"""Real2Sim -- tkinter GUI (analytical retargeting edition).

Layout:
  TOP    : camera preview  |  simulation preview (large, resizable)
  MID    : Start/Stop/Calibrate / camera selector / mirror toggle
  LEFT   : Settings (sliders: smoothing, visibility threshold, FPS)
  RIGHT  : Joint angles + scrollable log

Run:
    python scripts/run_gui.py [--camera IDX]

Deps: Pillow (pip install Pillow)
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
from real2sim.one_euro import OneEuroFilterArray
from real2sim.pose import PoseEstimator
from real2sim.retarget import compute_arm_angles, compute_debug_info
from real2sim.sim import G1Sim

try:
    from PIL import Image, ImageTk, ImageDraw
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

import tkinter as tk
from tkinter import ttk


# =============================================================================
# Communication structures
# =============================================================================

@dataclass
class SimFrame:
    cam_bgr:    Optional[np.ndarray]
    sim_rgb:    Optional[np.ndarray]
    fps:        float
    targets:    Optional[np.ndarray]   # (14,) joint angles
    calib:      Optional[BodyCalibration]
    status:     str
    debug_info: Optional[dict]        = None


@dataclass
class SimSettings:
    """Live-tunable settings -- writable from the GUI thread."""
    oef_min_cutoff:    float = config.OEF_MIN_CUTOFF
    oef_beta:          float = config.OEF_BETA
    visibility_thresh: float = config.VISIBILITY_THRESHOLD
    target_fps:        int   = config.TARGET_FPS
    debug_mode:        bool  = False


# =============================================================================
# Worker thread
# =============================================================================

class SimThread(threading.Thread):
    """Runs the camera → pose → OEF → retarget → physics loop."""

    def __init__(
        self,
        camera_idx: int,
        mirror:     bool,
        frame_q:    queue.Queue,
        log_q:      queue.Queue,
        settings:   SimSettings,
    ) -> None:
        super().__init__(daemon=True)
        self.camera_idx = camera_idx
        self.mirror     = mirror
        self.frame_q    = frame_q
        self.log_q      = log_q
        self.settings   = settings
        self._stop_evt  = threading.Event()
        self._calib_evt = threading.Event()
        self.error: Optional[str] = None

    def stop(self)                -> None: self._stop_evt.set()
    def request_calibration(self) -> None: self._calib_evt.set()

    def _log(self, msg: str) -> None:
        self.log_q.put(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _open_camera(self, idx: int):
        import cv2
        self._log(f"Opening camera (index={idx})...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera (index={idx})")
        return cap

    # -- Main loop --------------------------------------------------------------

    def run(self) -> None:  # noqa: C901
        import cv2

        cap  = None
        pose = None
        _csv_file = None
        _csv_writer = None

        try:
            cap = self._open_camera(self.camera_idx)

            self._log("Loading pose estimator...")
            pose = PoseEstimator()

            self._log("Loading G1 MuJoCo model...")
            sim      = G1Sim()
            renderer = sim.make_renderer(config.FRAME_W, config.FRAME_H)
            self._log(f"G1 loaded -- robot arm: {sim.arm_length_robot:.3f} m")

            calib = BodyCalibration.load(config.CALIBRATION_PATH)
            if calib is not None:
                self._log(
                    f"Calibration loaded -- scale={calib.scale:.3f}, "
                    f"user_arm={calib.arm_length_user:.3f} m"
                )
            else:
                self._log("No calibration cache -- press Calibrate (3 poses).")
            anatomical_zero = (
                np.asarray(calib.anatomical_zero, dtype=np.float64)
                if calib is not None else None
            )

            lm_filter = OneEuroFilterArray(shape=(33, 3))
            prev_oef  = (self.settings.oef_min_cutoff, self.settings.oef_beta)

            _cached_sim_pixels = None
            self._log("Ready.")

            t0           = time.time()
            frame_idx    = 0
            last_targets = sim.data.ctrl[sim.actuator_ids].copy()
            _targets_valid = False
            _t_prev      = time.time()

            while not self._stop_evt.is_set():
                s = self.settings

                _t_now = time.time()
                actual_dt = max(0.005, min(_t_now - _t_prev, 0.5))
                _t_prev = _t_now

                n_substeps = max(1, round(actual_dt / sim.dt))
                dt = actual_dt

                # -- Rebuild OEF if parameters changed --
                cur_oef = (s.oef_min_cutoff, s.oef_beta)
                if cur_oef != prev_oef:
                    lm_filter = OneEuroFilterArray(
                        shape=(33, 3),
                        min_cutoff=s.oef_min_cutoff, beta=s.oef_beta,
                    )
                    prev_oef = cur_oef
                    self._log(
                        f"OEF updated: min_cutoff={s.oef_min_cutoff:.2f}, "
                        f"beta={s.oef_beta:.3f}"
                    )

                # -- Calibration request --
                if self._calib_evt.is_set():
                    self._calib_evt.clear()
                    self._log("3-pose calibration starting...")

                    # Cache one rendered sim frame so the sim preview stays
                    # populated during the (~22 s) calibration.
                    renderer.update_scene(sim.data)
                    cached_sim_calib = renderer.render()

                    def _calib_frame_cb(bgr_image):
                        sf_calib = SimFrame(
                            cam_bgr    = bgr_image,
                            sim_rgb    = cached_sim_calib,
                            fps        = 0.0,
                            targets    = None,
                            calib      = None,
                            status     = "Calibrating",
                            debug_info = None,
                        )
                        try:
                            self.frame_q.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_q.put(sf_calib)

                    try:
                        calib = run_calibration(
                            cap, pose,
                            arm_length_robot=sim.arm_length_robot,
                            on_frame=_calib_frame_cb,
                            on_abort=lambda: self._stop_evt.is_set(),
                        )
                        calib.save()
                        anatomical_zero = np.asarray(
                            calib.anatomical_zero, dtype=np.float64,
                        )
                        self._log(
                            f"Calibration done -- scale={calib.scale:.3f}, "
                            f"shoulder_w={calib.shoulder_width_user:.3f} m, "
                            f"torso_h={calib.torso_height_user:.3f} m"
                        )
                    except RuntimeError as exc:
                        self._log(f"CALIBRATION ERROR: {exc}")

                # -- Camera read + pose --
                ret, frame = cap.read()
                if not ret:
                    self._log("ERROR: camera read failed!")
                    time.sleep(0.1)
                    continue
                res = pose.process(frame)

                # -- Retargeting --
                thr = s.visibility_thresh
                dbg = None
                if res.world_landmarks is not None:
                    vis = res.visibility
                    arms_visible = (
                        vis is not None
                        and all(vis[i] >= thr for i in (
                            config.LM_LEFT_SHOULDER,  config.LM_RIGHT_SHOULDER,
                            config.LM_LEFT_ELBOW,     config.LM_RIGHT_ELBOW,
                            config.LM_LEFT_WRIST,     config.LM_RIGHT_WRIST,
                            config.LM_LEFT_HIP,       config.LM_RIGHT_HIP,
                        ))
                    )
                    if arms_visible:
                        lms_smooth = lm_filter.update(res.world_landmarks, dt)
                        try:
                            last_targets = compute_arm_angles(
                                lms_smooth, anatomical_zero=anatomical_zero,
                            )
                            _targets_valid = True
                            # Always populate the live-debug dict so the GUI
                            # panel has data, calibrated or not.
                            dbg = compute_debug_info(
                                lms_smooth, anatomical_zero, res.visibility,
                            )
                        except Exception as exc:
                            self._log(f"Retarget error: {exc}")
                    elif frame_idx % 90 == 0:
                        self._log(f"Required landmarks below threshold {thr:.2f}")

                # -- Physics step --
                sim.step_smooth(
                    n_substeps=n_substeps,
                    ik_target=last_targets if _targets_valid else None,
                )

                # -- Render (throttled) --
                if frame_idx % _RENDER_EVERY == 0:
                    renderer.update_scene(sim.data)
                    _cached_sim_pixels = renderer.render()
                sim_pixels = _cached_sim_pixels

                cam_show = (
                    cv2.flip(res.annotated, 1) if self.mirror
                    else res.annotated.copy()
                )

                fps = (frame_idx + 1) / max(1e-6, time.time() - t0)

                # -- Debug overlay + CSV --
                if s.debug_mode and dbg is not None:
                    import csv as _csv
                    if _csv_file is None:
                        _csv_path = _ROOT / "debug_log.csv"
                        _csv_file = open(_csv_path, "w", newline="", encoding="utf-8")
                        _csv_writer = _csv.writer(_csv_file)
                        _csv_writer.writerow([
                            "frame", "t", "fps",
                            "L_up_x", "L_up_y", "L_up_z",
                            "L_fo_x", "L_fo_y", "L_fo_z",
                            "R_up_x", "R_up_y", "R_up_z",
                            "R_fo_x", "R_fo_y", "R_fo_z",
                        ] + [f"q_raw{i:02d}" for i in range(14)]
                          + [f"q_g1{i:02d}"  for i in range(14)])
                        self._log(f"Debug CSV: {_csv_path}")

                    Lu = dbg["L_upper_body"]; Lf = dbg["L_fore_body"]
                    Ru = dbg["R_upper_body"]; Rf = dbg["R_fore_body"]
                    _csv_writer.writerow([
                        frame_idx, f"{time.time()-t0:.3f}", f"{fps:.1f}",
                        f"{Lu[0]:.4f}", f"{Lu[1]:.4f}", f"{Lu[2]:.4f}",
                        f"{Lf[0]:.4f}", f"{Lf[1]:.4f}", f"{Lf[2]:.4f}",
                        f"{Ru[0]:.4f}", f"{Ru[1]:.4f}", f"{Ru[2]:.4f}",
                        f"{Rf[0]:.4f}", f"{Rf[1]:.4f}", f"{Rf[2]:.4f}",
                    ] + [f"{x:+.4f}" for x in dbg["q_raw"]]
                      + [f"{x:+.4f}" for x in dbg["q_g1_target"]])
                    if frame_idx % 10 == 0:
                        _csv_file.flush()
                elif not s.debug_mode and _csv_file is not None:
                    _csv_file.close()
                    _csv_file = None
                    _csv_writer = None
                    self._log("Debug CSV closed.")

                sf = SimFrame(
                    cam_bgr    = cam_show,
                    sim_rgb    = sim_pixels,
                    fps        = fps,
                    targets    = last_targets.copy(),
                    calib      = calib,
                    status     = "Running" if calib else "Not calibrated",
                    debug_info = dbg,
                )
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
                self.frame_q.put(sf)

                frame_idx += 1

                _dt_target = 1.0 / max(1, s.target_fps)
                elapsed = time.time() - t0
                expected = frame_idx * _dt_target
                sleep_t = expected - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except Exception as exc:
            import traceback
            self.error = traceback.format_exc()
            self._log(f"FATAL ERROR: {exc}")
        finally:
            if _csv_file is not None:
                try:
                    _csv_file.close()
                except Exception:
                    pass
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                if pose is not None:
                    pose.close()
            except Exception:
                pass
            self._log("Thread stopped.")


# =============================================================================
# Theme
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

POLL_MS = 33   # ~30 Hz GUI refresh

_RENDER_EVERY = 3   # render every 3rd frame to keep CPU happy


# =============================================================================
# Helpers
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
    img  = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    bb   = draw.textbbox((0, 0), text)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(80, 80, 110))
    return ImageTk.PhotoImage(img)


def _make_slider(parent, row, label, var, from_, to, resolution, fmt=".3f"):
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
        parent, variable=var, from_=from_, to=to, resolution=resolution,
        orient="horizontal",
        bg=BG2, fg=TEXT, troughcolor=BG3,
        activebackground=ACCENT, highlightthickness=0,
        sliderrelief="flat", showvalue=False, length=160,
        command=lambda v: val_lbl.configure(text=f"{float(v):{fmt}}"),
    )
    sl.grid(row=row, column=2, sticky="ew", padx=(0, 4), pady=2)
    return sl


# =============================================================================
# Main GUI
# =============================================================================

class Real2SimGUI:
    def __init__(self, root: tk.Tk, start_camera: int = config.CAMERA_INDEX) -> None:
        self.root     = root
        self._thread: Optional[SimThread] = None
        self._running = False

        self._mirror  = tk.BooleanVar(value=True)
        self._cam_idx = tk.IntVar(value=start_camera)

        self._settings = SimSettings()
        self._s_oef_cutoff = tk.DoubleVar(value=self._settings.oef_min_cutoff)
        self._s_oef_beta   = tk.DoubleVar(value=self._settings.oef_beta)
        self._s_vis_thresh = tk.DoubleVar(value=self._settings.visibility_thresh)
        self._s_fps        = tk.IntVar(value=self._settings.target_fps)
        self._s_debug      = tk.BooleanVar(value=False)

        self._frame_q: queue.Queue[SimFrame] = queue.Queue(maxsize=1)
        self._log_q:   queue.Queue[str]      = queue.Queue()

        root.title("Real2Sim -- Unitree G1 control")
        root.configure(bg=BG)
        root.resizable(True, True)
        root.minsize(1100, 700)

        self._build_ui()
        self._poll()

    # =========================================================================
    # UI build
    # =========================================================================

    def _build_ui(self) -> None:
        root = self.root
        # Previews vs. bottom panel: leave ~half the height for the bottom so
        # joint angles + debug + log all fit comfortably on one page.
        root.rowconfigure(0, weight=3)   # camera + sim previews
        root.rowconfigure(1, weight=0)   # control bar (fixed)
        root.rowconfigure(2, weight=3)   # joints + debug + log
        root.columnconfigure(0, weight=1)

        # Preview row
        self._preview_row = tk.Frame(root, bg=BG)
        self._preview_row.grid(row=0, column=0, sticky="nsew", padx=4, pady=(8, 2))
        self._preview_row.rowconfigure(0, weight=1)
        self._preview_row.columnconfigure(0, weight=1)
        self._preview_row.columnconfigure(1, weight=1)

        self._cam_frame = tk.Frame(self._preview_row, bg=BG2, bd=0)
        self._cam_frame.rowconfigure(1, weight=1)
        self._cam_frame.columnconfigure(0, weight=1)
        tk.Label(
            self._cam_frame, text="  Camera", bg=BG2, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._cam_lbl = tk.Label(self._cam_frame, bg="#000")
        self._cam_lbl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._cam_frame.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=2)

        self._sim_frame = tk.Frame(self._preview_row, bg=BG2, bd=0)
        self._sim_frame.rowconfigure(1, weight=1)
        self._sim_frame.columnconfigure(0, weight=1)
        tk.Label(
            self._sim_frame, text="  Simulation", bg=BG2, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._sim_lbl = tk.Label(self._sim_frame, bg="#000")
        self._sim_lbl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._sim_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=2)

        if _PIL_OK:
            for lbl, txt in (
                (self._cam_lbl, "camera not active"),
                (self._sim_lbl, "sim not running"),
            ):
                ph = _placeholder_img(480, 320, txt)
                lbl.configure(image=ph)
                lbl.image = ph

        # Control row
        ctrl = tk.Frame(root, bg=BG3, padx=8, pady=4)
        ctrl.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        self._start_btn = _btn(ctrl, "  Start ", GREEN, BG, self._on_start)
        self._start_btn.pack(side="left", padx=(0, 4))
        self._stop_btn  = _btn(ctrl, "  Stop  ", RED, BG, self._on_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=(0, 4))
        self._calib_btn = _btn(ctrl, "  Calibrate (3 poses) ", BG2, ACCENT,
                               self._on_calibrate, state="disabled")
        self._calib_btn.pack(side="left", padx=(0, 4))

        self._clear_calib_btn = _btn(
            ctrl, "  Clear calib ", BG2, ORANGE, self._on_clear_calibration,
        )
        self._clear_calib_btn.pack(side="left", padx=(0, 16))

        tk.Checkbutton(
            ctrl, text="Mirror", variable=self._mirror,
            bg=BG3, fg=TEXT, selectcolor=BG2,
            activebackground=BG3, activeforeground=ACCENT,
            font=("Segoe UI", 9),
            command=self._on_mirror_toggle,
        ).pack(side="left", padx=(0, 16))

        self._fps_var   = tk.StringVar(value="--")
        self._state_var = tk.StringVar(value="Stopped")
        self._scale_var = tk.StringVar(value="--")

        for lbl, var, col in (
            ("FPS:",    self._fps_var,   YELLOW),
            ("Scale:",  self._scale_var, TEXT),
            ("Status:", self._state_var, GREEN),
        ):
            tk.Label(ctrl, text=lbl, bg=BG3, fg=SUBTEXT,
                     font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
            tk.Label(ctrl, textvariable=var, bg=BG3, fg=col,
                     font=("Segoe UI", 9, "bold"), width=8).pack(side="left")

        tk.Label(ctrl, text="  |  Camera:", bg=BG3, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
        tk.Spinbox(
            ctrl, from_=0, to=9, width=3,
            textvariable=self._cam_idx,
            bg=BG2, fg=TEXT, buttonbackground=BG2,
            insertbackground=TEXT, relief="flat",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(2, 4))
        _btn(ctrl, "Apply camera", BG2, ORANGE, self._on_camera_apply).pack(
            side="left", padx=(8, 0),
        )

        # Bottom panels — 3 columns: Settings | Joints | Debug
        bottom = tk.Frame(root, bg=BG)
        bottom.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 8))
        bottom.columnconfigure(0, weight=0, minsize=300)   # settings
        bottom.columnconfigure(1, weight=0, minsize=270)   # joint angles
        bottom.columnconfigure(2, weight=1)                # live debug
        bottom.rowconfigure(0, weight=3)                   # main panels
        bottom.rowconfigure(1, weight=1)                   # log

        self._build_settings(bottom)
        self._build_joints(bottom)
        self._build_debug(bottom)
        self._build_log(bottom)

    # ── Settings panel ───────────────────────────────────────────────────────

    def _build_settings(self, parent: tk.Frame) -> None:
        sf = tk.LabelFrame(
            parent, text="  Settings  ",
            bg=BG2, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        sf.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(4, 2), pady=2)
        sf.columnconfigure(2, weight=1)

        row = 0
        tk.Label(sf, text="-- Smoothing (One-Euro) --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(6, 2))
        row += 1
        _make_slider(sf, row, "min_cutoff (Hz):",
                     self._s_oef_cutoff, 0.1, 5.0, 0.05, fmt=".2f")
        row += 1
        tk.Label(sf, text="low = smoother, more lag",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 4))
        row += 1
        _make_slider(sf, row, "beta (speed):",
                     self._s_oef_beta, 0.0, 0.5, 0.005, fmt=".3f")
        row += 1
        tk.Label(sf, text="higher = less lag on fast motion",
                 bg=BG2, fg=SUBTEXT, font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(0, 6))
        row += 1

        tk.Label(sf, text="-- Visibility threshold --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        _make_slider(sf, row, "visibility:",
                     self._s_vis_thresh, 0.1, 0.95, 0.05, fmt=".2f")
        row += 1

        tk.Label(sf, text="-- Pipeline FPS --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        _make_slider(sf, row, "target FPS:",
                     self._s_fps, 5, 60, 1, fmt=".0f")
        row += 1

        tk.Label(sf, text="-- Debug --",
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 8, "bold"),
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", padx=4, pady=(4, 2))
        row += 1
        tk.Checkbutton(
            sf, text="Debug CSV (debug_log.csv)",
            variable=self._s_debug,
            bg=BG2, fg=TEXT, selectcolor=BG3,
            activebackground=BG2, activeforeground=ACCENT,
            font=("Segoe UI", 8),
            command=self._on_apply_settings,
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))
        row += 1

        _btn(sf, "Apply settings", BG3, ACCENT, self._on_apply_settings,
             ).grid(row=row, column=0, columnspan=3, sticky="ew",
                    padx=4, pady=(4, 8))

    # ── Joint angles panel (bottom col 1) ───────────────────────────────────

    def _build_joints(self, parent: tk.Frame) -> None:
        jf = tk.LabelFrame(
            parent, text="  Joint angles (rad)  ",
            bg=BG, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        jf.grid(row=0, column=1, sticky="nsew", padx=(2, 2), pady=(2, 2))
        jf.columnconfigure(0, weight=1)
        jf.columnconfigure(1, weight=1)
        jf.rowconfigure(1, weight=1)

        self._joint_vars: list[tuple[tk.StringVar, ttk.Progressbar]] = []
        names = config.ARM_JOINT_NAMES
        half = len(names) // 2

        style = ttk.Style()
        style.theme_use("default")
        style.configure("J.Horizontal.TProgressbar",
                        background=ACCENT, troughcolor=BG2,
                        bordercolor=BG2, lightcolor=ACCENT, darkcolor=ACCENT)

        for side_idx, side_lbl in enumerate(("Left arm", "Right arm")):
            tk.Label(jf, text=side_lbl, bg=BG, fg=ACCENT,
                     font=("Segoe UI", 8, "bold"),
                     anchor="w").grid(row=0, column=side_idx,
                                      sticky="ew", padx=4, pady=(2, 0))
            col = tk.Frame(jf, bg=BG)
            col.grid(row=1, column=side_idx, sticky="nsew", padx=4, pady=2)
            for j in range(half):
                jname = names[side_idx * half + j]
                short = (jname.replace("left_", "")
                              .replace("right_", "")
                              .replace("_joint", "")
                              .replace("shoulder_", "sh_"))
                rf = tk.Frame(col, bg=BG)
                rf.pack(fill="x", pady=0)
                tk.Label(rf, text=short, bg=BG, fg=TEXT,
                         font=("Consolas", 8), width=12,
                         anchor="w").pack(side="left")
                vv = tk.StringVar(value=" 0.000")
                tk.Label(rf, textvariable=vv, bg=BG, fg=YELLOW,
                         font=("Consolas", 9, "bold"),
                         width=7).pack(side="left")
                bar = ttk.Progressbar(rf, length=70, mode="determinate",
                                      orient="horizontal",
                                      style="J.Horizontal.TProgressbar")
                bar.pack(side="left", padx=(2, 0))
                bar["value"] = 50
                self._joint_vars.append((vv, bar))

    # ── Live debug panel (bottom col 2) — 2-column Label grid, no scroll ────

    def _build_debug(self, parent: tk.Frame) -> None:
        df = tk.LabelFrame(
            parent, text="  Live debug (inputs → math → output)  ",
            bg=BG, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        df.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=(2, 2))
        df.columnconfigure(0, weight=1)
        df.columnconfigure(1, weight=1)
        df.rowconfigure(0, weight=1)

        # Left half: visibility + body vectors
        left = tk.Frame(df, bg=BG)
        left.grid(row=0, column=0, sticky="nw", padx=(4, 2), pady=4)

        # Right half: q_raw + anatomical_zero + q_g1_target
        right = tk.Frame(df, bg=BG)
        right.grid(row=0, column=1, sticky="nw", padx=(2, 4), pady=4)

        _H = ("Consolas", 8, "bold")
        _V = ("Consolas", 7)

        def _hdr(frame, row, text):
            tk.Label(frame, text=text, bg=BG, fg=ACCENT,
                     font=_H, anchor="w").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ── LEFT: INPUT visibility ───────────────────────────────────────────
        r = 0
        _hdr(left, r, "INPUT  visibility"); r += 1
        vis_names = ["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr", "L_hp", "R_hp"]
        self._dbg_vis_lbls: dict[str, tk.Label] = {}
        for row_offset, start in enumerate((0, 4)):
            row_f = tk.Frame(left, bg=BG)
            row_f.grid(row=r, column=0, columnspan=4, sticky="w"); r += 1
            for name in vis_names[start:start + 4]:
                tk.Label(row_f, text=name + ":", bg=BG, fg=SUBTEXT,
                         font=_V, anchor="w").pack(side="left")
                lbl = tk.Label(row_f, text="-.--", bg=BG, fg=TEXT,
                               font=_V, width=5, anchor="w")
                lbl.pack(side="left", padx=(0, 4))
                self._dbg_vis_lbls[name] = lbl

        # ── LEFT: MATH body frame vectors ────────────────────────────────────
        _hdr(left, r, "MATH   body frame vectors"); r += 1
        _vec_defs = [
            ("L_upper", "L_upper_body"),
            ("L_fore",  "L_fore_body"),
            ("R_upper", "R_upper_body"),
            ("R_fore",  "R_fore_body"),
        ]
        self._dbg_vec_vars: dict[str, tk.StringVar] = {}
        for name, key in _vec_defs:
            row_f = tk.Frame(left, bg=BG)
            row_f.grid(row=r, column=0, columnspan=4, sticky="w"); r += 1
            tk.Label(row_f, text=f"  {name:7}", bg=BG, fg=SUBTEXT,
                     font=_V, width=9, anchor="w").pack(side="left")
            sv = tk.StringVar(value="x=+0.00 y=+0.00 z=+0.00")
            tk.Label(row_f, textvariable=sv, bg=BG, fg=YELLOW,
                     font=_V).pack(side="left")
            self._dbg_vec_vars[key] = sv

        # ── RIGHT: MATH q_raw ────────────────────────────────────────────────
        r = 0
        _hdr(right, r, "MATH   q_raw"); r += 1
        self._dbg_qraw_lbls: list[tk.Label] = []
        self._dbg_qraw_strs: list[tk.StringVar] = []
        for arm in ("L", "R"):
            row_f = tk.Frame(right, bg=BG)
            row_f.grid(row=r, column=0, sticky="w"); r += 1
            tk.Label(row_f, text=f"  {arm}", bg=BG, fg=SUBTEXT,
                     font=_V, width=3, anchor="w").pack(side="left")
            sv = tk.StringVar(value="pitch=+0.00 roll=+0.00 elbow=+0.00")
            lbl = tk.Label(row_f, textvariable=sv, bg=BG, fg=YELLOW, font=_V)
            lbl.pack(side="left")
            self._dbg_qraw_lbls.append(lbl)
            self._dbg_qraw_strs.append(sv)

        # ── RIGHT: CALIB anatomical_zero ─────────────────────────────────────
        _hdr(right, r, "CALIB  anatomical_zero"); r += 1
        self._dbg_azero_lbls: list[tk.Label] = []
        self._dbg_azero_strs: list[tk.StringVar] = []
        for arm in ("L", "R"):
            row_f = tk.Frame(right, bg=BG)
            row_f.grid(row=r, column=0, sticky="w"); r += 1
            tk.Label(row_f, text=f"  {arm}", bg=BG, fg=SUBTEXT,
                     font=_V, width=3, anchor="w").pack(side="left")
            sv = tk.StringVar(value="pitch=+0.00 roll=+0.00 elbow=+0.00")
            lbl = tk.Label(row_f, textvariable=sv, bg=BG, fg=SUBTEXT, font=_V)
            lbl.pack(side="left")
            self._dbg_azero_lbls.append(lbl)
            self._dbg_azero_strs.append(sv)

        # ── RIGHT: OUTPUT q_g1_target ────────────────────────────────────────
        _hdr(right, r, "OUTPUT q_g1_target"); r += 1
        self._dbg_qg1_lbls: list[tk.Label] = []
        self._dbg_qg1_strs: list[tk.StringVar] = []
        for arm in ("L", "R"):
            row_f = tk.Frame(right, bg=BG)
            row_f.grid(row=r, column=0, sticky="w"); r += 1
            tk.Label(row_f, text=f"  {arm}", bg=BG, fg=SUBTEXT,
                     font=_V, width=3, anchor="w").pack(side="left")
            sv = tk.StringVar(value="pitch=+0.00 roll=+0.00 elbow=+0.00")
            lbl = tk.Label(row_f, textvariable=sv, bg=BG, fg=GREEN, font=_V)
            lbl.pack(side="left")
            self._dbg_qg1_lbls.append(lbl)
            self._dbg_qg1_strs.append(sv)

    # ── Log panel (bottom row 1, col 1+2) ───────────────────────────────────

    def _build_log(self, parent: tk.Frame) -> None:
        lf = tk.LabelFrame(
            parent, text="  Log  ",
            bg=BG, fg=ACCENT, font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        lf.grid(row=1, column=1, columnspan=2, sticky="nsew",
                padx=(2, 4), pady=(0, 2))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        self._log_text = tk.Text(
            lf, bg=BG2, fg=TEXT,
            font=("Consolas", 8),
            state="disabled", wrap="word",
            relief="flat", height=4,
        )
        self._log_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, command=self._log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._log_text["yscrollcommand"] = sb.set

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_start(self) -> None:
        if self._running:
            return
        self._apply_settings_to_struct()
        self._start_thread()

    def _start_thread(self) -> None:
        cam = self._cam_idx.get()
        self._frame_q = queue.Queue(maxsize=1)
        self._log_q   = queue.Queue()
        self._thread  = SimThread(
            camera_idx=cam,
            mirror=self._mirror.get(),
            frame_q=self._frame_q,
            log_q=self._log_q,
            settings=self._settings,
        )
        self._thread.start()
        self._running = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._calib_btn.configure(state="normal")
        self._state_var.set("Running")
        self._log_append(
            f"[{time.strftime('%H:%M:%S')}] Thread started (camera={cam})",
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
        self._state_var.set("Stopped")
        self._fps_var.set("--")
        self._scale_var.set("--")
        if _PIL_OK:
            for lbl, txt in (
                (self._cam_lbl, "camera not active"),
                (self._sim_lbl, "sim not running"),
            ):
                ph = _placeholder_img(480, 320, txt)
                lbl.configure(image=ph)
                lbl.image = ph
        self._log_append(f"[{time.strftime('%H:%M:%S')}] Thread stopped.")

    def _on_calibrate(self) -> None:
        if self._thread is not None:
            self._thread.request_calibration()
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Calibration requested -- "
                "hold T-pose first, then A-pose, then elbows 90°."
            )

    def _on_clear_calibration(self) -> None:
        """Delete the cached anatomical_zero so the live pipeline uses the
        RAW math (no per-user offset). Useful for debugging — lets you see
        whether the math itself produces sensible angles, independent of any
        calibration bias."""
        p = config.CALIBRATION_PATH
        if p.exists():
            try:
                p.unlink()
                self._log_append(
                    f"[{time.strftime('%H:%M:%S')}] Cleared {p.name} -- "
                    "restart pipeline (Stop then Start) to use raw angles."
                )
            except OSError as exc:
                self._log_append(f"Could not delete {p}: {exc}")
        else:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] No calibration cache to clear."
            )

    def _on_camera_apply(self) -> None:
        new_cam = self._cam_idx.get()
        if self._running:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Switching to camera={new_cam} "
                "(restarting thread...)"
            )
            self._on_stop()
            self.root.after(300, self._start_thread)
        else:
            self._log_append(
                f"[{time.strftime('%H:%M:%S')}] Camera={new_cam} set (Start to begin)"
            )

    def _on_mirror_toggle(self) -> None:
        if self._thread is not None:
            self._thread.mirror = self._mirror.get()

    def _apply_settings_to_struct(self) -> None:
        self._settings.oef_min_cutoff    = float(self._s_oef_cutoff.get())
        self._settings.oef_beta          = float(self._s_oef_beta.get())
        self._settings.visibility_thresh = float(self._s_vis_thresh.get())
        self._settings.target_fps        = int(self._s_fps.get())
        self._settings.debug_mode        = bool(self._s_debug.get())

    def _on_apply_settings(self) -> None:
        self._apply_settings_to_struct()
        s = self._settings
        self._log_append(
            f"[{time.strftime('%H:%M:%S')}] Settings: "
            f"cutoff={s.oef_min_cutoff:.2f}  beta={s.oef_beta:.3f}  "
            f"vis={s.visibility_thresh:.2f}  fps={s.target_fps}"
        )

    def _on_close(self) -> None:
        if self._running:
            self._on_stop()
        self.root.destroy()

    # =========================================================================
    # Log + image update
    # =========================================================================

    def _log_append(self, msg: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _update_preview(self, label: tk.Label, arr: np.ndarray, bgr: bool) -> None:
        if not _PIL_OK:
            return
        lw = label.winfo_width()
        lh = label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = 480, 320

        import cv2 as _cv
        rgb = _cv.cvtColor(arr, _cv.COLOR_BGR2RGB) if bgr else arr

        ih, iw = rgb.shape[:2]
        scale  = min(lw / iw, lh / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img    = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)

        canvas = Image.new("RGB", (lw, lh), (10, 10, 20))
        ox = (lw - nw) // 2
        oy = (lh - nh) // 2
        canvas.paste(img, (ox, oy))

        photo = ImageTk.PhotoImage(canvas)
        label.configure(image=photo, text="")
        label.image = photo

    def _update_joints(self, targets: np.ndarray) -> None:
        PI = 3.14159
        for i, (vv, bar) in enumerate(self._joint_vars):
            a = float(targets[i])
            vv.set(f"{a:+.3f}")
            bar["value"] = min(100.0, max(0.0, (a / PI + 1.0) * 50.0))

    # -- Live debug panel (label-based, no scroll) ---------------------------

    def _update_debug(self, dbg: Optional[dict]) -> None:
        """Push the latest debug dict into the static label grid."""
        _DASH = "-.--"
        _ZERO = "pitch=+0.00 roll=+0.00 elbow=+0.00"

        if dbg is None:
            for lbl in self._dbg_vis_lbls.values():
                lbl.configure(text=_DASH, fg=SUBTEXT)
            for sv in self._dbg_vec_vars.values():
                sv.set("x=+0.00 y=+0.00 z=+0.00")
            for sv, lbl in zip(self._dbg_qraw_strs, self._dbg_qraw_lbls):
                sv.set(_ZERO); lbl.configure(fg=SUBTEXT)
            for sv, lbl in zip(self._dbg_azero_strs, self._dbg_azero_lbls):
                sv.set(_ZERO); lbl.configure(fg=SUBTEXT)
            for sv, lbl in zip(self._dbg_qg1_strs, self._dbg_qg1_lbls):
                sv.set(_ZERO); lbl.configure(fg=SUBTEXT)
            return

        # Visibility
        vis = dbg.get("visibility", {})
        for name, lbl in self._dbg_vis_lbls.items():
            v = vis.get(name)
            if v is None:
                lbl.configure(text=_DASH, fg=SUBTEXT)
            else:
                fg = GREEN if v >= 0.8 else (YELLOW if v >= 0.5 else RED)
                lbl.configure(text=f"{v:.2f}", fg=fg)

        # Body frame vectors
        for key, sv in self._dbg_vec_vars.items():
            if key in dbg:
                x, y, z = dbg[key]
                sv.set(f"x={x:+.2f} y={y:+.2f} z={z:+.2f}")

        # q_raw (anatomical)
        if "q_raw" in dbg:
            q = dbg["q_raw"]
            for (pi_i, ro_i, el_i), sv, lbl in zip(
                ((0, 1, 3), (7, 8, 10)),
                self._dbg_qraw_strs, self._dbg_qraw_lbls,
            ):
                sv.set(f"pitch={q[pi_i]:+.2f} roll={q[ro_i]:+.2f} elbow={q[el_i]:+.2f}")
                lbl.configure(fg=YELLOW)

        # anatomical_zero
        if "anatomical_zero" in dbg:
            z = dbg["anatomical_zero"]
            nonzero = any(abs(v) > 1e-3 for v in z)
            for (pi_i, ro_i, el_i), sv, lbl in zip(
                ((0, 1, 3), (7, 8, 10)),
                self._dbg_azero_strs, self._dbg_azero_lbls,
            ):
                sv.set(f"pitch={z[pi_i]:+.2f} roll={z[ro_i]:+.2f} elbow={z[el_i]:+.2f}")
                lbl.configure(fg=TEXT if nonzero else SUBTEXT)

        # q_g1_target
        if "q_g1_target" in dbg:
            q = dbg["q_g1_target"]
            for (pi_i, ro_i, el_i), sv, lbl in zip(
                ((0, 1, 3), (7, 8, 10)),
                self._dbg_qg1_strs, self._dbg_qg1_lbls,
            ):
                sv.set(f"pitch={q[pi_i]:+.2f} roll={q[ro_i]:+.2f} elbow={q[el_i]:+.2f}")
                lbl.configure(fg=GREEN)

    def _poll(self) -> None:
        try:
            while True:
                self._log_append(self._log_q.get_nowait())
        except queue.Empty:
            pass

        if (self._thread is not None
                and not self._thread.is_alive()
                and self._running):
            err = self._thread.error or "(unknown error)"
            self._log_append(f"ERROR -- thread crashed:\n{err}")
            self._running = False
            self._thread  = None
            self._start_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._calib_btn.configure(state="disabled")
            self._state_var.set("ERROR")

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
            self._update_debug(latest.debug_info)
            if _PIL_OK:
                if latest.cam_bgr is not None:
                    self._update_preview(self._cam_lbl, latest.cam_bgr, bgr=True)
                if latest.sim_rgb is not None:
                    self._update_preview(self._sim_lbl, latest.sim_rgb, bgr=False)

        self.root.after(POLL_MS, self._poll)


# =============================================================================
# Entry
# =============================================================================

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Real2Sim GUI")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = p.parse_args()

    if not _PIL_OK:
        print(
            "WARNING: Pillow missing -- preview disabled.\n"
            "pip install Pillow",
            file=sys.stderr,
        )

    root = tk.Tk()
    app  = Real2SimGUI(root, start_camera=args.camera)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
