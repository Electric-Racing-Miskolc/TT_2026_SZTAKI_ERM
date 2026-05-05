"""Pose-mode szintetizalo segedfuggvenyek a Real2Sim-hez.

Harom mod tamogatott:
  MODE_3D     : 1 kamera, MediaPipe 3D becsles (default, eredeti viselkedes)
  MODE_2D     : 1 kamera, frontalis sikra projektalt (forward/back nullazas)
  MODE_FUSION : 2 kamera (front + side @90 jobb), tengely-szerinti fuzio

Mind a 2D, mind a fusion mod ugy mukodik, hogy szintetikus (33, 3) world-landmark
tombot allit elo, amit az ArmIK valtozatlan formaban kepes feldolgozni. A wrist
landmarkokat ugy konstruaaljuk, hogy a body-frame transzformacio utan a kivant
rel_body komponenseket adjak vissza:

    rel_body_target = [rel_body_x_chosen, rel_body_y_chosen, rel_body_z_chosen]
    synth_wrist     = mp_shoulder + R_body @ rel_body_target

Ekkor (lms_synth[wr] - lms_synth[sh]) = R_body @ rel_body_target, es
R_body.T @ (...) = rel_body_target -- pontosan, amit szeretnenk.
"""

from __future__ import annotations

import numpy as np

from . import config
from .ik import build_body_frame

# ── Mode konstansok ──────────────────────────────────────────────────────────
MODE_3D     = "single_3d"
MODE_2D     = "single_2d"
MODE_FUSION = "dual_fusion"

MODES = (MODE_3D, MODE_2D, MODE_FUSION)

MODE_LABELS = {
    MODE_3D:     "1 kamera (3D)",
    MODE_2D:     "1 kamera (2D, sik mozgas)",
    MODE_FUSION: "2 kamera (fuzio)",
}

# (oldal, shoulder_idx, wrist_idx) parok
_ARM_PAIRS = (
    ("left",  config.LM_LEFT_SHOULDER,  config.LM_LEFT_WRIST),
    ("right", config.LM_RIGHT_SHOULDER, config.LM_RIGHT_WRIST),
)


# ── Mode B: 1 kamera, 2D projekcio ───────────────────────────────────────────

def make_2d_landmarks(lms_front: np.ndarray) -> np.ndarray:
    """Mode B: zero out the forward/back component (rel_body[1]) for both wrists.

    A felhasznalo karjat a sajat frontalis sikjara projektaljuk
    (bal-jobb + fel-le), elore-hatra mozgas elhanyagolva.

    Parameters
    ----------
    lms_front : (33, 3) float array
        Raw vagy szuretelt MediaPipe world landmarks a front kamerabol.

    Returns
    -------
    (33, 3) float array -- szintetikus landmarks az IK szamara.
    """
    R   = build_body_frame(lms_front)
    out = lms_front.copy()
    for _arm, sh_idx, wr_idx in _ARM_PAIRS:
        rel = R.T @ (lms_front[wr_idx] - lms_front[sh_idx])
        rel[1] = 0.0                                # forward/back nullazas
        out[wr_idx] = lms_front[sh_idx] + R @ rel
    return out


# ── Mode C: 2 kamera, tengely-szerinti fuzio ─────────────────────────────────

def fuse_dual_landmarks(
    lms_front:    np.ndarray,
    lms_side:     np.ndarray | None,
    last_y_state: dict[str, float],
    side_ok:      bool,
) -> np.ndarray:
    """Mode C: 2 kamerabol szarmazo landmarks fuzioja.

    A felhasznalo body-frame koordinataiban:
      rel_body[0] (bal-jobb)        <- FRONT kamera (kepsikban megbizhato)
      rel_body[1] (elore-hatra)     <- SIDE  kamera (oldalrol kepsikban)
      rel_body[2] (fel-le)          <- FRONT kamera (kepsikban)

    Ha side_ok=False (side kamera elveszti a tracket), az utolso jol meresett
    rel_body[1] erteket hasznaljuk minden karra (freeze).

    Parameters
    ----------
    lms_front : (33, 3) float array
        Front kamera szuretelt world landmarks.
    lms_side : (33, 3) float array vagy None
        Side kamera szuretelt world landmarks (None ha nincs detektalva).
    last_y_state : dict[str, float]
        Szal-elettartamu allapot {'left': float, 'right': float}; a fuggveny
        in-place frissiti, ha side_ok=True.
    side_ok : bool
        True, ha a side kamera erveny0 detekciot adott; False eseten freeze.

    Returns
    -------
    (33, 3) float array -- szintetikus landmarks az IK szamara.
    """
    R_front = build_body_frame(lms_front)
    out     = lms_front.copy()
    R_side  = (
        build_body_frame(lms_side)
        if (side_ok and lms_side is not None) else None
    )

    for arm, sh_idx, wr_idx in _ARM_PAIRS:
        # Front kamera rel_body
        rel_f = R_front.T @ (lms_front[wr_idx] - lms_front[sh_idx])

        # Side kamera rel_body (vagy utolso jo ertek)
        if side_ok and R_side is not None and lms_side is not None:
            rel_s = R_side.T @ (lms_side[wr_idx] - lms_side[sh_idx])
            last_y_state[arm] = float(rel_s[1])
        rel_y = last_y_state.get(arm, 0.0)

        # Fuzio: [front_X, side_Y, front_Z]
        rel_fused = np.array([rel_f[0], rel_y, rel_f[2]], dtype=np.float64)

        # Vissza a kamera-koordinatakba a front body-frame-en keresztul
        out[wr_idx] = lms_front[sh_idx] + R_front @ rel_fused

    return out


# ── Lathatosagi segitfuggveny ────────────────────────────────────────────────

def arms_visible(visibility: np.ndarray | None, threshold: float) -> bool:
    """True, ha a vall+wrist landmarkok mind a kuszob folott."""
    if visibility is None:
        return False
    required = (
        config.LM_LEFT_SHOULDER,  config.LM_RIGHT_SHOULDER,
        config.LM_LEFT_WRIST,     config.LM_RIGHT_WRIST,
    )
    return all(visibility[i] >= threshold for i in required)
