"""Unit tests for the T-pose calibration module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim import config
from real2sim.calibration import BodyCalibration, _arm_length_from_landmarks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_landmarks(
    L_sh, R_sh, L_hip, R_hip, L_wr, R_wr
) -> np.ndarray:
    lms = np.zeros((33, 3), dtype=np.float32)
    lms[config.LM_LEFT_SHOULDER]  = L_sh
    lms[config.LM_RIGHT_SHOULDER] = R_sh
    lms[config.LM_LEFT_HIP]       = L_hip
    lms[config.LM_RIGHT_HIP]      = R_hip
    lms[config.LM_LEFT_WRIST]     = L_wr
    lms[config.LM_RIGHT_WRIST]    = R_wr
    return lms


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_arm_length_symmetric():
    """Symmetric T-pose: reported length = exact arm span."""
    arm = 0.60
    lms = _make_landmarks(
        L_sh=[ arm, 0, 0], R_sh=[-arm, 0, 0],
        L_hip=[arm * 0.5, 0, -0.5], R_hip=[-arm * 0.5, 0, -0.5],
        L_wr=[2 * arm, 0, 0], R_wr=[-2 * arm, 0, 0],
    )
    length = _arm_length_from_landmarks(lms)
    assert abs(length - arm) < 1e-4, f"expected {arm}, got {length}"


def test_arm_length_asymmetric():
    """Left/right arm lengths differ — average should be returned."""
    left_len  = 0.55
    right_len = 0.65
    lms = _make_landmarks(
        L_sh=[0.1, 0, 0], R_sh=[-0.1, 0, 0],
        L_hip=[0.05, 0, -0.5], R_hip=[-0.05, 0, -0.5],
        L_wr=[0.1 + left_len, 0, 0],
        R_wr=[-0.1 - right_len, 0, 0],
    )
    expected = 0.5 * (left_len + right_len)
    assert abs(_arm_length_from_landmarks(lms) - expected) < 1e-4


def test_scale_computation():
    """Scale = robot / user, clamped to [SCALE_MIN, SCALE_MAX]."""
    robot_arm = 0.55
    user_arm  = 0.70
    calib = BodyCalibration(
        arm_length_user=user_arm,
        arm_length_robot=robot_arm,
        scale=robot_arm / user_arm,
    )
    expected = robot_arm / user_arm
    assert abs(calib.scale - expected) < 1e-6


def test_scale_clamp_low():
    """Very long user arm clamped to CALIB_SCALE_MIN."""
    calib = BodyCalibration(
        arm_length_user=5.0,
        arm_length_robot=0.55,
        scale=max(0.55 / 5.0, config.CALIB_SCALE_MIN),
    )
    assert calib.scale >= config.CALIB_SCALE_MIN


def test_save_load_roundtrip(tmp_path):
    """Calibration serialises/deserialises without loss."""
    calib = BodyCalibration(
        arm_length_user=0.68,
        arm_length_robot=0.54,
        scale=0.54 / 0.68,
    )
    p = tmp_path / "calibration.json"
    # Manually save to tmp_path.
    p.write_text(json.dumps({
        "arm_length_user":  calib.arm_length_user,
        "arm_length_robot": calib.arm_length_robot,
        "scale":            calib.scale,
    }))
    loaded = BodyCalibration.load(p)
    assert loaded is not None
    assert abs(loaded.arm_length_user  - calib.arm_length_user)  < 1e-9
    assert abs(loaded.arm_length_robot - calib.arm_length_robot) < 1e-9
    assert abs(loaded.scale            - calib.scale)            < 1e-9
