"""Real2Sim -- tkinter GUI

Layout:
  TOP    : kamera preview  |  szimulacios preview  (nagy, nyujthato)
  KOZEP  : Start/Stop/Kalibr. / kamera valaszto / tukrozes
  BALRA  : Beallitasok (csuszkok: smoothness, IK, lathatosagi kuszob)
  JOBBRA : Izuleti szogek + gorgethet log

Indithas:
    python scripts/run_gui.py [--camera IDX]

Fugg: Pillow  (pip install Pillow)
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from real2sim import config
from real2sim.calibration import BodyCalibration, run_calibration
from real2sim.fusion import (
    MODE_3D, MODE_2D, MODE_FUSION, MODES, MODE_LABELS,
    make_2d_landmarks, fuse_dual_landmarks, arms_visible,
)
from real2sim.ik import ArmIK
from real2sim.one_euro import OneEuroFilterArray
from real2sim.pose import PoseEstimator
from real2sim.sim import G1Sim

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

import tkinter as tk
from tkinter import ttk


# =============================================================================
# Kommunikacios adatstrukturak
# =============================================================================

@dataclass
class SimFrame:
    cam_bgr:      Optional[np.ndarray]
    sim_rgb:      Optional[np.ndarray]
    fps:          float
    targets:      Optional[np.ndarray]   # (14,) izuleti szogek
    calib:        Optional[BodyCalibration]
    status:       str
    cam_side_bgr: Optional[np.ndarray] = None   # csak Mode C-ben
    debug_info:   Optional[dict]        = None   # debug adatok


@dataclass
class SimSettings:
    """Elo beallitasok -- GUI-bol barmikorhivhato."""
    oef_min_cutoff:      float = config.OEF_MIN_CUTOFF
    oef_beta:            float = config.OEF_BETA
    visibility_thresh:   float = config.VISIBILITY_THRESHOLD
    ik_posture_cost:     float = config.IK_POSTURE_COST
    target_fps:          int   = config.TARGET_FPS
    debug_mode:          bool  = False
    # Mode + kamerak
    mode:                str   = MODE_3D
    side_camera_idx:     int   = 0           # csak Mode C-ben hasznalt


# =============================================================================
# Hatterthrad
# =============================================================================

class SimThread(threading.Thread):
    """Futtatja a kamera -> pose -> OEF -> [mode-szintezis] -> IK -> fizika ciklust.

    Modok (settings.mode):
        MODE_3D     : 1 kamera, MediaPipe 3D becsles (default)
        MODE_2D     : 1 kamera, frontalis sikra projektalt
        MODE_FUSION : 2 kamera (front + side @90 jobb), tengely-szerinti fuzio
    """

    def __init__(
        self,
        camera_idx:      int,
        mirror:          bool,
        frame_q:         queue.Queue,
        log_q:           queue.Queue,
        settings:        SimSettings,
        side_camera_idx: int = 0,
    ) -> None:
        super().__init__(daemon=True)
        self.camera_idx      = camera_idx       # front kamera
        self.side_camera_idx = side_camera_idx  # csak Mode C-ben
        self.mirror          = mirror
        self.frame_q         = frame_q
        self.log_q           = log_q
        self.settings        = settings
        self._stop_evt       = threading.Event()
        self._calib_evt      = threading.Event()
        self.error: Optional[str] = None

    def stop(self)                -> None: self._stop_evt.set()
    def request_calibration(self) -> None: self._calib_evt.set()

    def _log(self, msg: str) -> None:
        self.log_q.put(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # -- Segit metodusok ------------------------------------------------------

    def _open_camera(self, idx: int, label: str):
        import cv2
        self._log(f"{label} kamera megnyitasa (index={idx})...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        if not cap.isOpened():
            raise RuntimeError(f"Nem nyilik meg a {label} kamera (index={idx})")
        return cap

    # -- Fo ciklus ------------------------------------------------------------

    def run(self) -> None:  # noqa: C901
        import cv2

        cap_front = None
        cap_side  = None
        pose_front = None
        pose_side  = None

        try:
            mode = self.settings.mode
            self._log(f"Mod: {MODE_LABELS.get(mode, mode)}")

            # -- Kamerak --
            cap_front = self._open_camera(self.camera_idx, "Front")

            if mode == MODE_FUSION:
                if self.side_camera_idx == self.camera_idx:
                    raise RuntimeError(
                        f"A side es a front kamera index nem lehet azonos "
                        f"(mindketto = {self.camera_idx})"
                    )
                cap_side = self._open_camera(self.side_camera_idx, "Side")
                self._log("FIGYELEM: Mode C-ben az FPS feleződhet "
                          "(2 pose-becsles soros). Csokkentsd a target FPS-t ha akadozik.")

            # -- Pose estimator(ok) --
            self._log("Pose estimator betoltese...")
            pose_front = PoseEstimator()
            if mode == MODE_FUSION:
                pose_side = PoseEstimator()

            # -- Szimulacios modell --
            self._log("G1 MuJoCo modell betoltese...")
            sim      = G1Sim()
            renderer = sim.make_renderer(config.FRAME_W, config.FRAME_H)
            self._log(f"G1 betoltve -- robot kar: {sim.arm_length_robot:.3f} m")

            # -- Kalibraciot betoltese cache-bol --
            calib = BodyCalibration.load(config.CALIBRATION_PATH)
            if calib is not None:
                self._log(
                    f"Kalibraciot betoltve -- scale={calib.scale:.3f}, "
                    f"felhasznaloi kar={calib.arm_length_user:.3f} m"
                )
            else:
                self._log("Nincs kalibraciot -- nyomd meg a Kalibraciot gombot!")

            # -- IK + OEF szurok --
            self._log("IK inicializalasa...")
            arm_ik         = ArmIK(sim)
            lm_filter_f    = OneEuroFilterArray(shape=(33, 3))
            lm_filter_s    = (
                OneEuroFilterArray(shape=(33, 3)) if mode == MODE_FUSION else None
            )
            prev_oef          = (self.settings.oef_min_cutoff, self.settings.oef_beta)
            prev_posture_cost = self.settings.ik_posture_cost

            # Fusion mode allapot: utolso jol meresett rel_body[1] freezehez
            # Kulcsok: "left_wrist", "right_wrist", "left_elbow", "right_elbow"
            last_y_state = {
                "left_wrist":  0.0, "right_wrist":  0.0,
                "left_elbow":  0.0, "right_elbow":  0.0,
            }
            _cached_sim_pixels = None
            self._log("Keszen all.")

            t0           = time.time()
            frame_idx    = 0
            # Initialise ctrl targets from the current stand-pose ctrl values
            # so step_smooth() starts from the correct resting position.
            last_targets = sim.data.ctrl[sim.actuator_ids].copy()
            _ik_valid    = False          # True once the first IK result arrives
            _t_prev      = time.time()   # wall-clock time of previous frame
            _csv_file    = None
            _csv_writer  = None

            while not self._stop_evt.is_set():
                s = self.settings

                # -- Actual elapsed time since previous frame --
                _t_now   = time.time()
                actual_dt = max(0.005, min(_t_now - _t_prev, 0.5))
                _t_prev  = _t_now

                # n_substeps computed from real elapsed time so physics runs
                # at wall-clock speed regardless of MediaPipe latency.
                n_substeps = max(1, round(actual_dt / sim.dt))

                # dt for OEF filter = actual inter-frame interval
                dt = actual_dt

                # -- OEF ujraepitese ha megvaltoztak a parameterek --
                cur_oef = (s.oef_min_cutoff, s.oef_beta)
                if cur_oef != prev_oef:
                    lm_filter_f = OneEuroFilterArray(
                        shape=(33, 3),
                        min_cutoff=s.oef_min_cutoff, beta=s.oef_beta,
                    )
                    if mode == MODE_FUSION:
                        lm_filter_s = OneEuroFilterArray(
                            shape=(33, 3),
                            min_cutoff=s.oef_min_cutoff, beta=s.oef_beta,
                        )
                    prev_oef = cur_oef
                    self._log(
                        f"OEF frissitve: min_cutoff={s.oef_min_cutoff:.2f}, "
                        f"beta={s.oef_beta:.3f}"
                    )

                # -- IK posture cost frissitese (csak valtozaskor) --
                if s.ik_posture_cost != prev_posture_cost:
                    try:
                        arm_ik._posture_task.set_cost(s.ik_posture_cost)
                        prev_posture_cost = s.ik_posture_cost
                        self._log(f"IK posture cost frissitve: {s.ik_posture_cost:.4f}")
                    except Exception as exc:
                        self._log(f"posture cost frissites sikertelen: {exc}")

                # -- Ujrakalibralasi kerest (mindig front kamerabol) --
                if self._calib_evt.is_set():
                    self._calib_evt.clear()
                    self._log("T-poz kalibraciot indul... (tartsd ki a karjaidat!)")
                    try:
                        calib = run_calibration(
                            cap_front, pose_front,
                            arm_length_robot=sim.arm_length_robot,
                        )
                        calib.save()
                        self._log(
                            f"Kalibraciot kesz -- scale={calib.scale:.3f}, "
                            f"felhasznaloi kar={calib.arm_length_user:.3f} m"
                        )
                    except RuntimeError as exc:
                        self._log(f"KALIBRACIOT HIBA: {exc}")

                # -- Front kamera olvasas + pose --
                ret_f, frame_f = cap_front.read()
                if not ret_f:
                    self._log("HIBA: Front kamera olvasas sikertelen!")
                    time.sleep(0.1)
                    continue
                res_f = pose_front.process(frame_f)

                # -- Side kamera olvasas + pose (csak Mode C) --
                res_s = None
                cam_side_show = None
                if mode == MODE_FUSION and cap_side is not None and pose_side is not None:
                    ret_s, frame_s = cap_side.read()
                    if ret_s:
                        res_s = pose_side.process(frame_s)
                        cam_side_show = (
                            cv2.flip(res_s.annotated, 1) if self.mirror
                            else res_s.annotated.copy()
                        )
                    elif frame_idx % 90 == 0:
                        self._log("FIGYELEM: side kamera olvasas sikertelen.")

                # -- IK pipeline --
                thr = s.visibility_thresh
                if calib is not None and res_f.world_landmarks is not None:
                    front_ok = arms_visible(res_f.visibility, thr)

                    if front_ok:
                        # Front kamera szureles
                        lms_f = lm_filter_f.update(res_f.world_landmarks, dt)

                        # Mode-szerinti szintezis
                        try:
                            if mode == MODE_3D:
                                synth = lms_f
                            elif mode == MODE_2D:
                                synth = make_2d_landmarks(lms_f)
                            elif mode == MODE_FUSION:
                                side_ok = (
                                    res_s is not None
                                    and res_s.world_landmarks is not None
                                    and arms_visible(res_s.visibility, thr)
                                )
                                lms_s = (
                                    lm_filter_s.update(res_s.world_landmarks, dt)
                                    if side_ok and lm_filter_s is not None
                                    else None
                                )
                                synth = fuse_dual_landmarks(
                                    lms_f, lms_s, last_y_state, side_ok=side_ok,
                                )
                                if not side_ok and frame_idx % 90 == 0:
                                    self._log(
                                        "Side kamera nem lathato karok -- "
                                        "freeze az utolso elore-hatra erteken."
                                    )
                            else:
                                synth = lms_f

                            last_targets = arm_ik.step(synth, calib, dt=dt, visibility=res_f.visibility)
                            _ik_valid = True
                        except Exception as exc:
                            self._log(f"IK lepes hiba: {exc}")

                    elif frame_idx % 90 == 0:
                        vis = res_f.visibility
                        if vis is not None:
                            vL_sh = vis[config.LM_LEFT_SHOULDER]
                            vR_sh = vis[config.LM_RIGHT_SHOULDER]
                            vL_wr = vis[config.LM_LEFT_WRIST]
                            vR_wr = vis[config.LM_RIGHT_WRIST]
                            self._log(
                                f"Front: karok nem lathatoak (kuszob={thr:.2f}): "
                                f"L_sh={vL_sh:.2f} R_sh={vR_sh:.2f} "
                                f"L_wr={vL_wr:.2f} R_wr={vR_wr:.2f}"
                            )
                        else:
                            self._log("Front: nincs lathatosagi adat (pose nem detektalt)")

                # -- Fizika: valós idejű step + per-substep ctrl simítás --
                sim.step_smooth(
                    n_substeps=n_substeps,
                    ik_target=last_targets if _ik_valid else None,
                )

                # -- Render (throttled: csak minden _RENDER_EVERY-edik frame-ben) --
                if frame_idx % _RENDER_EVERY == 0:
                    renderer.update_scene(sim.data)
                    _cached_sim_pixels = renderer.render()
                sim_pixels = _cached_sim_pixels

                cam_show = (
                    cv2.flip(res_f.annotated, 1) if self.mirror
                    else res_f.annotated.copy()
                )

                fps = (frame_idx + 1) / max(1e-6, time.time() - t0)
                dbg = arm_ik.last_debug if arm_ik.last_debug else None

                # -- Debug overlay + CSV logging --
                if s.debug_mode and dbg:
                    import csv as _csv
                    from pathlib import Path as _Path

                    # CSV megnyitasa (ha meg nincs)
                    if _csv_file is None:
                        _csv_path = _Path(__file__).resolve().parent.parent / "debug_log.csv"
                        _csv_file = open(_csv_path, "w", newline="", encoding="utf-8")
                        _csv_writer = _csv.writer(_csv_file)
                        _csv_writer.writerow([
                            "frame", "t", "mode", "fps",
                            "L_body_x", "L_body_y", "L_body_z",
                            "R_body_x", "R_body_y", "R_body_z",
                            "L_tgt_x",  "R_tgt_x",
                        ])
                        self._log(f"Debug CSV: {_csv_path}")

                    Lb = dbg["L_body"]
                    Rb = dbg["R_body"]
                    _csv_writer.writerow([
                        frame_idx, f"{time.time()-t0:.3f}", mode, f"{fps:.1f}",
                        f"{Lb[0]:.4f}", f"{Lb[1]:.4f}", f"{Lb[2]:.4f}",
                        f"{Rb[0]:.4f}", f"{Rb[1]:.4f}", f"{Rb[2]:.4f}",
                        f"{dbg['L_tgt_x']:.4f}", f"{dbg['R_tgt_x']:.4f}",
                    ])
                    if frame_idx % 10 == 0:
                        _csv_file.flush()

                    # Kamera kep overlay
                    oy = cam_show.shape[0] - 10
                    for txt, col in [
                        (f"MODE: {mode}  FPS:{fps:.1f}", (200, 200, 50)),
                        (f"L body  x={Lb[0]:+.3f}  y={Lb[1]:+.3f}  z={Lb[2]:+.3f}", (100, 220, 100)),
                        (f"R body  x={Rb[0]:+.3f}  y={Rb[1]:+.3f}  z={Rb[2]:+.3f}", (100, 180, 255)),
                        (f"L tgt_X={dbg['L_tgt_x']:+.3f}  R tgt_X={dbg['R_tgt_x']:+.3f}", (220, 180, 100)),
                    ]:
                        oy -= 20
                        cv2.putText(cam_show, txt, (6, oy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

                elif not s.debug_mode and _csv_file is not None:
                    _csv_file.close()
                    _csv_file   = None
                    _csv_writer = None
                    self._log("Debug CSV lezarva.")

                sf = SimFrame(
                    cam_bgr      = cam_show,
                    sim_rgb      = sim_pixels,
                    fps          = fps,
                    targets      = last_targets.copy(),
                    calib        = calib,
                    status       = "Fut" if calib else "Kalibralatlan",
                    cam_side_bgr = cam_side_show,
                    debug_info   = dbg,
                )
                # Csak a legfrissebb frame-t tartjuk
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
                self.frame_q.put(sf)

                frame_idx += 1

                # -- Wall-clock pacing (fixed rate, independent of actual_dt) --
                _dt_target = 1.0 / max(1, s.target_fps)
                elapsed    = time.time() - t0
                expected   = frame_idx * _dt_target
                sleep_t    = expected - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except Exception as exc:
            import traceback
            self.error = traceback.format_exc()
            self._log(f"FATALIS HIBA: {exc}")
        finally:
            if _csv_file is not None:
                try:
                    _csv_file.close()
                except Exception:
                    pass
            for cap in (cap_front, cap_side):
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
            for pe in (pose_front, pose_side):
                try:
                    if pe is not None:
                        pe.close()
                except Exception:
                    pass
            self._log("Szal leallitva.")


# =============================================================================
# Szinpaletta (sotet tema)
# =============================================================================
BG      = "#1e1e2e"
BG2     = "#2a2a3d"
BG3     = "#313244"
ACCENT  = "#89b4fa"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
ORANGE  = "#fab387"
TEXT    = "#cdd6f4"
SUBTEXT = "#6c7086"

POLL_MS = 33   # ~30 Hz GUI frissites

# Offscreen renderer: eredeti felbontas, de csak minden 3. frame-ben renderel
_RENDER_EVERY = 3   # display ~10 fps, fizika 30 fps


# =============================================================================
# Segitfuggvenyek
# =============================================================================

def _btn(parent, text, bg, fg, cmd, **kw):
    return tk.Button(
        parent, text=text, bg=bg, fg=fg,
        relief="flat", padx=10, pady=5,
        font=("Segoe UI", 9, "bold"),
        cursor="hand2",
        activebackground=ACCENT, activeforeground=BG,
        command=cmd, **kw,
    )


def _placeholder_img(w: int, h: int, text: str):
    """Sotet hatter, kozepre irt szoveggel."""
    img  = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    bb   = draw.textbbox((0, 0), text)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(80, 80, 110))
    return ImageTk.PhotoImage(img)


# =============================================================================
# Csuszko sor segitmetodus
# =============================================================================

def _make_slider(
    parent, row, label, var: tk.DoubleVar,
    from_, to, resolution,
    fmt=".3f", orient="horizontal",
):
    """Egysoros csuszko: [cimke]  [ertek]  [------slider------]"""
    tk.Label(
        parent, text=label, bg=BG2, fg=TEXT,
        font=("Segoe UI", 8), anchor="w", width=20,
    ).grid(row=row, column=0, sticky="w", padx=(4, 0), pady=2)

    val_lbl = tk.Label(
        parent, text=f"{var.get():{fmt}}", bg=BG2, fg=YELLOW,
        font=("Consolas", 8, "bold"), width=7, anchor="e",
    )
    val_lbl.grid(row=row, column=1, padx=(2, 4), pady=2)

    sl = tk.Scale(
        parent,
        variable=var, from_=from_, to=to, resolution=resolution,
        orient=orient,
        bg=BG2, fg=TEXT, troughcolor=BG3,
        activebackground=ACCENT,
        highlightthickness=0,
        sliderrelief="flat",
        showvalue=False,
        length=160,
        command=lambda v: val_lbl.configure(text=f"{float(v):{fmt}}"),
    )
    sl.grid(row=row, column=2, sticky="ew", padx=(0, 4), pady=2)
    return sl


# =============================================================================
# Fo GUI osztaly
# =============================================================================

class Real2SimGUI:
    def __init__(self, root: tk.Tk, start_camera: int = config.CAMERA_INDEX) -> None:
        self.root     = root
        self._thread: Optional[SimThread] = None
        self._running = False

        self._mirror       = tk.BooleanVar(value=True)
        self._cam_idx      = tk.IntVar(value=start_camera)
        self._side_cam_idx = tk.IntVar(value=0)
        self._mode_var     = tk.StringVar(value=MODE_3D)

        # -- Elo beallitasok (csuszkok altal vezerlve) --
        self._settings = SimSettings()
        self._s_oef_cutoff  = tk.DoubleVar(value=self._settings.oef_min_cutoff)
        self._s_oef_beta    = tk.DoubleVar(value=self._settings.oef_beta)
        self._s_vis_thresh  = tk.DoubleVar(value=self._settings.visibility_thresh)
        self._s_ik_posture  = tk.DoubleVar(value=self._settings.ik_posture_cost)
        self._s_fps         = tk.IntVar(value=self._settings.target_fps)
        self._s_debug       = tk.BooleanVar(value=False)

        self._frame_q: queue.Queue[SimFrame] = queue.Queue(maxsize=1)
        self._log_q:   queue.Queue[str]      = queue.Queue()

        root.title("Real2Sim -- Unitree G1 vezerlohely")
        root.configure(bg=BG)
        root.resizable(True, True)
        root.minsize(1100, 700)

        self._build_ui()
        self._poll()

    # =========================================================================
    # UI epites
    # =========================================================================

    def _build_ui(self) -> None:
        root = self.root

        # Foablak racsozata:
        #  sor 0 : preview-k (nagy, nyujthato)
        #  sor 1 : vezerlok (fix)
        #  sor 2 : beallitasok | izuletek + log
        root.rowconfigure(0, weight=5)    # preview-k: 5 resz
        root.rowconfigure(1, weight=0)    # vezerlok: fix
        root.rowconfigure(2, weight=2)    # also panel: 2 resz
        root.columnconfigure(0, weight=1)

        # ── 0. sor: preview-k ────────────────────────────────────────────────
        self._preview_row = tk.Frame(root, bg=BG)
        self._preview_row.grid(row=0, column=0, sticky="nsew", padx=4, pady=(8, 2))

        # Front kamera preview
        self._cam_frame = tk.Frame(self._preview_row, bg=BG2, bd=0)
        self._cam_frame.rowconfigure(1, weight=1)
        self._cam_frame.columnconfigure(0, weight=1)
        self._cam_title = tk.Label(
            self._cam_frame, text="  Kamera", bg=BG2, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), anchor="w",
        )
        self._cam_title.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._cam_lbl = tk.Label(self._cam_frame, bg="#000")
        self._cam_lbl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Side kamera preview (csak Mode C-ben latszik)
        self._side_cam_frame = tk.Frame(self._preview_row, bg=BG2, bd=0)
        self._side_cam_frame.rowconfigure(1, weight=1)
        self._side_cam_frame.columnconfigure(0, weight=1)
        tk.Label(
            self._side_cam_frame, text="  Side kamera (90 jobb)",
            bg=BG2, fg=ORANGE, font=("Segoe UI", 10, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._side_cam_lbl = tk.Label(self._side_cam_frame, bg="#000")
        self._side_cam_lbl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Szimulacios preview
        self._sim_frame = tk.Frame(self._preview_row, bg=BG2, bd=0)
        self._sim_frame.rowconfigure(1, weight=1)
        self._sim_frame.columnconfigure(0, weight=1)
        tk.Label(
            self._sim_frame, text="  Szimulacios", bg=BG2, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._sim_lbl = tk.Label(self._sim_frame, bg="#000")
        self._sim_lbl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        if _PIL_OK:
            for lbl, txt in (
                (self._cam_lbl,      "front nem aktiv"),
                (self._side_cam_lbl, "side nem aktiv"),
                (self._sim_lbl,      "sim nem fut"),
            ):
                ph = _placeholder_img(480, 320, txt)
                lbl.configure(image=ph)
                lbl.image = ph

        # Alap layout (Mode A/B): cam | sim
        self._apply_mode_layout(MODE_3D)

        # ── 1. sor: vezerlok (2 alsor) ───────────────────────────────────────
        ctrl_outer = tk.Frame(root, bg=BG3, padx=8, pady=4)
        ctrl_outer.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        # 1.A alsor: Start/Stop/Kalibraciot + statusz
        ctrl = tk.Frame(ctrl_outer, bg=BG3)
        ctrl.pack(fill="x", pady=(0, 2))

        self._start_btn = _btn(ctrl, "  Start ", GREEN, BG, self._on_start)
        self._start_btn.pack(side="left", padx=(0, 4))

        self._stop_btn = _btn(ctrl, "  Stop  ", RED, BG, self._on_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=(0, 4))

        self._calib_btn = _btn(ctrl, "  Kalibraciot (T-poz) ", BG2, ACCENT,
                               self._on_calibrate, state="disabled")
        self._calib_btn.pack(side="left", padx=(0, 16))

        tk.Checkbutton(
            ctrl, text="Tukrozos",
            variable=self._mirror,
            bg=BG3, fg=TEXT, selectcolor=BG2,
            activebackground=BG3, activeforeground=ACCENT,
            font=("Segoe UI", 9),
            command=self._on_mirror_toggle,
        ).pack(side="left", padx=(0, 16))

        # Statusz jobb oldal
        self._fps_var   = tk.StringVar(value="--")
        self._state_var = tk.StringVar(value="Leallva")
        self._scale_var = tk.StringVar(value="--")
        self._cam_info  = tk.StringVar(value=f"kamera: {self._cam_idx.get()}")

        for lbl, var, col in (
            ("FPS:",    self._fps_var,   YELLOW),
            ("Scale:",  self._scale_var, TEXT),
            ("Statusz:",self._state_var, GREEN),
        ):
            tk.Label(ctrl, text=lbl, bg=BG3, fg=SUBTEXT,
                     font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
            tk.Label(ctrl, textvariable=var, bg=BG3, fg=col,
                     font=("Segoe UI", 9, "bold"), width=8).pack(side="left")

        # 1.B alsor: Mod + kamerak
        ctrl2 = tk.Frame(ctrl_outer, bg=BG3)
        ctrl2.pack(fill="x", pady=(2, 0))

        tk.Label(ctrl2, text="Mod:", bg=BG3, fg=TEXT,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))

        for mode_const, label_text in (
            (MODE_3D,     "1 kam (3D)"),
            (MODE_2D,     "1 kam (2D, sik)"),
            (MODE_FUSION, "2 kam (fuzio)"),
        ):
            tk.Radiobutton(
                ctrl2, text=label_text,
                variable=self._mode_var, value=mode_const,
                bg=BG3, fg=TEXT, selectcolor=BG2,
                activebackground=BG3, activeforeground=ACCENT,
                font=("Segoe UI", 9),
                command=self._on_mode_change,
            ).pack(side="left", padx=(0, 6))

        tk.Label(ctrl2, text="  |  Front:", bg=BG3, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
        tk.Spinbox(
            ctrl2, from_=0, to=9, width=3,
            textvariable=self._cam_idx,
            bg=BG2, fg=TEXT, buttonbackground=BG2,
            insertbackground=TEXT, relief="flat",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(2, 4))

        # Side kamera spinbox + label (csak Mode C-ben latszik)
        self._side_cam_lbl_w = tk.Label(
            ctrl2, text="Side:", bg=BG3, fg=ORANGE,
            font=("Segoe UI", 9, "bold"),
        )
        self._side_cam_lbl_w.pack(side="left", padx=(8, 2))
        self._side_cam_spin = tk.Spinbox(
            ctrl2, from_=0, to=9, width=3,
            textvariable=self._side_cam_idx,
            bg=BG2, fg=TEXT, buttonbackground=BG2,
            insertbackground=TEXT, relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        self._side_cam_spin.pack(side="left", padx=(2, 4))

        self._cam_apply_btn = _btn(
            ctrl2, "Valtas (kamera/mod)", BG2, ORANGE, self._on_camera_apply,
        )
        self._cam_apply_btn.pack(side="left", padx=(8, 0))

        # Mode A/B-ben elrejtjuk a side kamera widget-eket
        self._update_side_cam_widget_visibility()

        # ── 2. sor: also panel ───────────────────────────────────────────────
        bottom = tk.Frame(root, bg=BG)
        bottom.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 8))
        bottom.columnconfigure(0, weight=0, minsize=320)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        # -- Bal: beallitasok --
        self._build_settings(bottom)

        # -- Jobb: izuletek + log --
        self._build_joints_log(bottom)

    # ── Beallitasok panel ─────────────────────────────────────────────────────

    def _build_settings(self, parent: tk.Frame) -> None:
        sf = tk.LabelFrame(
            parent, text="  Beallitasok  ",
            bg=BG2, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        sf.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=2)
        sf.columnconfigure(2, weight=1)

        row = 0

        # ---- One-Euro filter ----
        tk.Label(sf, text="-- Simitas (One-Euro filter) --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(6, 2))
        row += 1

        _make_slider(sf, row,
                     "min_cutoff  (Hz):",
                     self._s_oef_cutoff, 0.1, 5.0, 0.05, fmt=".2f")
        row += 1
        tk.Label(sf, text="alacsony = simabb, tobb lag",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 4))
        row += 1

        _make_slider(sf, row,
                     "beta  (sebesseg):",
                     self._s_oef_beta, 0.0, 0.5, 0.005, fmt=".3f")
        row += 1
        tk.Label(sf, text="magasabb = kevesebb lag gyors mozgasnal",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 6))
        row += 1

        # ---- Lathatosagi kuszob ----
        tk.Label(sf, text="-- Lathatosagi kuszob --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        _make_slider(sf, row,
                     "visibility  thresh:",
                     self._s_vis_thresh, 0.1, 0.95, 0.05, fmt=".2f")
        row += 1
        tk.Label(sf, text="alacsony = lazabb, magasabb = szigorabb",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 6))
        row += 1

        # ---- IK ----
        tk.Label(sf, text="-- IK beallitasok --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        _make_slider(sf, row,
                     "posture cost:",
                     self._s_ik_posture, 1e-4, 0.05, 1e-4, fmt=".4f")
        row += 1
        tk.Label(sf, text="magasabb = inkabb alap-poz, alacsony = lazabb",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 6))
        row += 1

        # ---- FPS ----
        tk.Label(sf, text="-- Pipeline FPS --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        _make_slider(sf, row,
                     "cel FPS:",
                     self._s_fps, 5, 60, 1, fmt=".0f")
        row += 1
        tk.Label(sf, text="magasabb = folyekonoabb, de tobb CPU",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 6))
        row += 1

        # ---- Debug mod ----
        tk.Label(sf, text="-- Debug mod --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        tk.Checkbutton(
            sf, text="Debug overlay + CSV log (debug_log.csv)",
            variable=self._s_debug,
            bg=BG2, fg=TEXT, selectcolor=BG3,
            activebackground=BG2, activeforeground=ACCENT,
            font=("Segoe UI", 8),
            command=self._on_apply_settings,
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))
        row += 1

        # Alkalmaz gomb
        _btn(
            sf, "Beallitasok alkalmazasa", BG3, ACCENT,
            self._on_apply_settings,
        ).grid(row=row, column=0, columnspan=3, sticky="ew",
               padx=4, pady=(4, 8))

    # ── Izuleti szogek + log panel ───────────────────────────────────────────

    def _build_joints_log(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=2)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)

        # -- Izuleti szogek --
        jf = tk.LabelFrame(
            right, text="  Izuleti szogek (rad)  ",
            bg=BG, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        jf.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        jf.columnconfigure(0, weight=1)
        jf.columnconfigure(1, weight=1)

        self._joint_vars: list[tuple[tk.StringVar, ttk.Progressbar]] = []
        names = config.ARM_JOINT_NAMES
        half  = len(names) // 2

        style = ttk.Style()
        style.theme_use("default")
        style.configure("J.Horizontal.TProgressbar",
                        background=ACCENT, troughcolor=BG2,
                        bordercolor=BG2, lightcolor=ACCENT, darkcolor=ACCENT)

        for side_idx, side_lbl in enumerate(("Bal kar", "Jobb kar")):
            col = tk.Frame(jf, bg=BG)
            col.grid(row=0, column=side_idx, sticky="nsew", padx=4, pady=4)
            tk.Label(col, text=side_lbl, bg=BG, fg=ACCENT,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            for j in range(half):
                jname = names[side_idx * half + j]
                short = (jname
                         .replace("left_", "L ")
                         .replace("right_", "R ")
                         .replace("_joint", ""))
                rf = tk.Frame(col, bg=BG)
                rf.pack(fill="x", pady=1)
                tk.Label(rf, text=f"{short:<22}", bg=BG, fg=TEXT,
                         font=("Consolas", 8), width=22,
                         anchor="w").pack(side="left")
                vv = tk.StringVar(value=" 0.000")
                tk.Label(rf, textvariable=vv, bg=BG, fg=YELLOW,
                         font=("Consolas", 9, "bold"),
                         width=7).pack(side="left")
                bar = ttk.Progressbar(rf, length=90, mode="determinate",
                                      orient="horizontal",
                                      style="J.Horizontal.TProgressbar")
                bar.pack(side="left", padx=(4, 0))
                bar["value"] = 50
                self._joint_vars.append((vv, bar))

        # -- Log --
        lf = tk.LabelFrame(
            right, text="  Log  ",
            bg=BG, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        lf.grid(row=1, column=0, sticky="nsew")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self._log_text = tk.Text(
            lf, bg=BG2, fg=TEXT,
            font=("Consolas", 8),
            state="disabled", wrap="word",
            relief="flat", height=6,
        )
        self._log_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, command=self._log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._log_text["yscrollcommand"] = sb.set

    # =========================================================================
    # Gomb visszahivasok
    # =========================================================================

    # ── Mode / layout-kezelo metodusok ────────────────────────────────────────

    def _apply_mode_layout(self, mode: str) -> None:
        """Atrendezi a preview-ket a kivalasztott mode szerint.

        Mode A/B: cam | sim   (1 sor, 2 oszlop)
        Mode C  : cam | side  (felso sor, 2 oszlop)
                  sim spans   (also sor, colspan=2)
        """
        pr = self._preview_row
        # Hide all
        for fr in (self._cam_frame, self._side_cam_frame, self._sim_frame):
            fr.grid_forget()

        # Reset weights
        for r in (0, 1):
            pr.rowconfigure(r, weight=0)
        for c in (0, 1):
            pr.columnconfigure(c, weight=0)

        if mode == MODE_FUSION:
            # 2x2 grid: 2 cams felul, sim alul colspan=2
            pr.rowconfigure(0, weight=1)
            pr.rowconfigure(1, weight=1)
            pr.columnconfigure(0, weight=1)
            pr.columnconfigure(1, weight=1)
            self._cam_frame.grid     (row=0, column=0, sticky="nsew", padx=(4, 2), pady=(2, 2))
            self._side_cam_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=(2, 2))
            self._sim_frame.grid     (row=1, column=0, columnspan=2,
                                      sticky="nsew", padx=4, pady=(2, 2))
        else:
            # 1 sor, 2 oszlop: cam | sim
            pr.rowconfigure(0, weight=1)
            pr.columnconfigure(0, weight=1)
            pr.columnconfigure(1, weight=1)
            self._cam_frame.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=2)
            self._sim_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=2)

    def _update_side_cam_widget_visibility(self) -> None:
        """Show/hide side cam spinbox + label based on mode."""
        is_fusion = (self._mode_var.get() == MODE_FUSION)
        if is_fusion:
            # Helyezzuk el a 'Valtas' gomb ELE -- az mindig packelve van
            self._side_cam_lbl_w.pack(side="left", padx=(8, 2),
                                      before=self._cam_apply_btn)
            self._side_cam_spin.pack(side="left", padx=(2, 4),
                                     before=self._cam_apply_btn)
        else:
            self._side_cam_lbl_w.pack_forget()
            self._side_cam_spin.pack_forget()

    def _on_mode_change(self) -> None:
        """Mode-radio megnyomasa eseten: layout valtas + thread restart."""
        new_mode = self._mode_var.get()
        self._apply_mode_layout(new_mode)
        self._update_side_cam_widget_visibility()
        self._log_append(
            f"[{time.strftime('%H:%M:%S')}] Mode valtas -> "
            f"{MODE_LABELS.get(new_mode, new_mode)}"
        )
        self._apply_settings_to_struct()   # settings.mode frissitese az uj modra
        if self._running:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Szal ujrainditasa az uj moddal..."
            )
            self._on_stop()
            self.root.after(300, self._start_thread_with_current)

    # ── Start / Stop / Kalibracio ─────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._running:
            return
        self._apply_settings_to_struct()
        self._start_thread_with_current()

    def _start_thread_with_current(self) -> None:
        """Thread inditas az aktualis (mar settingsbe atmasolt) mode-dal."""
        cam_front = self._cam_idx.get()
        cam_side  = self._side_cam_idx.get()
        mode      = self._settings.mode

        if mode == MODE_FUSION and cam_front == cam_side:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] HIBA: Mode C-ben a front es "
                f"side kamera index nem lehet azonos (mindketto = {cam_front})."
            )
            return

        self._frame_q = queue.Queue(maxsize=1)
        self._log_q   = queue.Queue()
        self._thread  = SimThread(
            camera_idx      = cam_front,
            mirror          = self._mirror.get(),
            frame_q         = self._frame_q,
            log_q           = self._log_q,
            settings        = self._settings,
            side_camera_idx = cam_side,
        )
        self._thread.start()
        self._running = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._calib_btn.configure(state="normal")
        self._state_var.set("Fut")

        if mode == MODE_FUSION:
            self._cam_info.set(f"front={cam_front}, side={cam_side}")
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Szal inditva mode={MODE_LABELS[mode]} "
                f"(front={cam_front}, side={cam_side})"
            )
        else:
            self._cam_info.set(f"kamera: {cam_front}")
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Szal inditva mode={MODE_LABELS[mode]} "
                f"(kamera={cam_front})"
            )

    def _on_stop(self) -> None:
        if not self._running or self._thread is None:
            return
        self._thread.stop()
        self._thread.join(timeout=6.0)
        self._running = False
        self._thread  = None
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._calib_btn.configure(state="disabled")
        self._state_var.set("Leallva")
        self._fps_var.set("--")
        self._scale_var.set("--")
        if _PIL_OK:
            for lbl, txt in (
                (self._cam_lbl,      "front nem aktiv"),
                (self._side_cam_lbl, "side nem aktiv"),
                (self._sim_lbl,      "sim nem fut"),
            ):
                ph = _placeholder_img(480, 320, txt)
                lbl.configure(image=ph)
                lbl.image = ph
        self._log_append(f"[{time.strftime('%H:%M:%S')}] Szal leallitva.")

    def _on_calibrate(self) -> None:
        if self._thread is not None:
            self._thread.request_calibration()
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Kalibraciot kerest elkuldve -- "
                "tartsd T-pozban a karjaidat!"
            )

    def _on_camera_apply(self) -> None:
        """Kamera/Mode index valtas: ha fut, ujrainditja a szalat."""
        new_front = self._cam_idx.get()
        new_side  = self._side_cam_idx.get()
        mode      = self._mode_var.get()
        self._apply_settings_to_struct()
        if self._running:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Valtas -> "
                f"front={new_front}, side={new_side}, mode={MODE_LABELS.get(mode, mode)} "
                "(ujrainditom a szalat...)"
            )
            self._on_stop()
            self.root.after(300, self._start_thread_with_current)
        else:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Beallitva: front={new_front}, "
                f"side={new_side}, mode={MODE_LABELS.get(mode, mode)} "
                "(Start-tal indul)"
            )

    def _on_mirror_toggle(self) -> None:
        if self._thread is not None:
            self._thread.mirror = self._mirror.get()

    def _apply_settings_to_struct(self) -> None:
        self._settings.oef_min_cutoff    = float(self._s_oef_cutoff.get())
        self._settings.oef_beta          = float(self._s_oef_beta.get())
        self._settings.visibility_thresh = float(self._s_vis_thresh.get())
        self._settings.ik_posture_cost   = float(self._s_ik_posture.get())
        self._settings.target_fps        = int(self._s_fps.get())
        self._settings.mode              = self._mode_var.get()
        self._settings.side_camera_idx   = int(self._side_cam_idx.get())
        self._settings.debug_mode        = bool(self._s_debug.get())

    def _on_apply_settings(self) -> None:
        self._apply_settings_to_struct()
        # SimThread kozvetlenul olvassa a settings objektumot,
        # tehat az azonnal ervenyes
        s = self._settings
        self._log_append(
            f"[{time.strftime('%H:%M:%S')}] Beallitasok alkalmazva: "
            f"cutoff={s.oef_min_cutoff:.2f}  beta={s.oef_beta:.3f}  "
            f"vis={s.visibility_thresh:.2f}  "
            f"ik_cost={s.ik_posture_cost:.4f}  "
            f"fps={s.target_fps}"
        )

    def _on_close(self) -> None:
        if self._running:
            self._on_stop()
        self.root.destroy()

    # =========================================================================
    # Log
    # =========================================================================

    def _log_append(self, msg: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # =========================================================================
    # Kepfrissites (PIL)
    # =========================================================================

    def _update_preview(self, label: tk.Label, arr: np.ndarray, bgr: bool) -> None:
        """Atmeretezetett kepet jeleniti meg a label-en, kitoltve a rendelkezesre allo helyet."""
        if not _PIL_OK:
            return
        lw = label.winfo_width()
        lh = label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = 480, 320

        import cv2 as _cv
        if bgr:
            rgb = _cv.cvtColor(arr, _cv.COLOR_BGR2RGB)
        else:
            rgb = arr

        # Arany-megtarto fit
        ih, iw = rgb.shape[:2]
        scale  = min(lw / iw, lh / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img    = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)

        # Fekete hatter kozepre
        canvas = Image.new("RGB", (lw, lh), (10, 10, 20))
        ox = (lw - nw) // 2
        oy = (lh - nh) // 2
        canvas.paste(img, (ox, oy))

        photo = ImageTk.PhotoImage(canvas)
        label.configure(image=photo, text="")
        label.image = photo

    # =========================================================================
    # Izuleti szog frissites
    # =========================================================================

    def _update_joints(self, targets: np.ndarray) -> None:
        PI = 3.14159
        for i, (vv, bar) in enumerate(self._joint_vars):
            a = float(targets[i])
            vv.set(f"{a:+.3f}")
            bar["value"] = min(100.0, max(0.0, (a / PI + 1.0) * 50.0))

    # =========================================================================
    # Poll -- fo szalban, minden POLL_MS ms
    # =========================================================================

    def _poll(self) -> None:
        # Log sorok
        try:
            while True:
                self._log_append(self._log_q.get_nowait())
        except queue.Empty:
            pass

        # Szal-osszeomlas
        if (self._thread is not None
                and not self._thread.is_alive()
                and self._running):
            err = self._thread.error or "(ismeretlen hiba)"
            self._log_append(f"HIBA -- szal leallitva:\n{err}")
            self._running = False
            self._thread  = None
            self._start_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._calib_btn.configure(state="disabled")
            self._state_var.set("HIBA")

        # Legfrissebb frame
        latest: Optional[SimFrame] = None
        try:
            while True:
                latest = self._frame_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._fps_var.set(f"{latest.fps:.1f}")
            self._state_var.set(latest.status)
            if latest.calib is not None:
                self._scale_var.set(f"{latest.calib.scale:.3f}")

            if latest.targets is not None:
                self._update_joints(latest.targets)

            if _PIL_OK:
                if latest.cam_bgr is not None:
                    self._update_preview(self._cam_lbl, latest.cam_bgr, bgr=True)
                if latest.sim_rgb is not None:
                    self._update_preview(self._sim_lbl, latest.sim_rgb, bgr=False)
                if latest.cam_side_bgr is not None:
                    self._update_preview(
                        self._side_cam_lbl, latest.cam_side_bgr, bgr=True,
                    )

        self.root.after(POLL_MS, self._poll)


# =============================================================================
# Belepes
# =============================================================================

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Real2Sim GUI")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = p.parse_args()

    if not _PIL_OK:
        print(
            "FIGYELEM: Pillow hianzik -- kepmegjelenitoes nem elerheto.\n"
            "pip install Pillow",
            file=sys.stderr,
        )

    root = tk.Tk()
    app  = Real2SimGUI(root, start_camera=args.camera)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
