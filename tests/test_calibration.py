"""Unit tests for the 3-pose anthropometric calibration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim import config
from real2sim.calibration import (
    BodyCalibration,
    _arm_length,
    _shoulder_width,
    _torso_height,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lms(**kw) -> np.ndarray:
    lms = np.zeros((33, 3), dtype=np.float32)
    for name, val in kw.items():
        idx = getattr(config, f"LM_{name.upper()}")
        lms[idx] = val
    return lms


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_arm_length_symmetric():
    arm = 0.60
    lms = _lms(
        left_shoulder=[ arm, 0, 0], right_shoulder=[-arm, 0, 0],
        left_hip=[ arm * 0.5, 0, -0.5], right_hip=[-arm * 0.5, 0, -0.5],
        left_wrist=[2 * arm, 0, 0], right_wrist=[-2 * arm, 0, 0],
    )
    assert abs(_arm_length(lms) - arm) < 1e-4


def test_arm_length_asymmetric():
    left_len, right_len = 0.55, 0.65
    lms = _lms(
        left_shoulder=[0.1, 0, 0], right_shoulder=[-0.1, 0, 0],
        left_hip=[0.05, 0, -0.5], right_hip=[-0.05, 0, -0.5],
        left_wrist=[0.1 + left_len, 0, 0],
        right_wrist=[-0.1 - right_len, 0, 0],
    )
    expected = 0.5 * (left_len + right_len)
    assert abs(_arm_length(lms) - expected) < 1e-4


def test_shoulder_width():
    lms = _lms(
        left_shoulder=[ 0.22, 0, 0], right_shoulder=[-0.22, 0, 0],
    )
    assert abs(_shoulder_width(lms) - 0.44) < 1e-5


def test_torso_height():
    lms = _lms(
        left_shoulder=[ 0.20, 0, 0.35], right_shoulder=[-0.20, 0, 0.35],
        left_hip=[ 0.10, 0, -0.15], right_hip=[-0.10, 0, -0.15],
    )
    # mid_shoulder.z = 0.35, mid_hip.z = -0.15, distance = 0.50
    assert abs(_torso_height(lms) - 0.50) < 1e-5


def test_save_load_roundtrip(tmp_path):
    calib = BodyCalibration(
        arm_length_user=0.68,
        shoulder_width_user=0.42,
        torso_height_user=0.55,
        arm_length_robot=0.54,
        scale=0.54 / 0.68,
        anatomical_zero=[0.01, -0.02, 0.0, 0.05] + [0.0] * 10,
    )
    p = tmp_path / "anthropo.json"
    calib.save(p)
    loaded = BodyCalibration.load(p)
    assert loaded is not None
    assert abs(loaded.arm_length_user     - calib.arm_length_user)     < 1e-9
    assert abs(loaded.shoulder_width_user - calib.shoulder_width_user) < 1e-9
    assert abs(loaded.torso_height_user   - calib.torso_height_user)   < 1e-9
    assert abs(loaded.scale               - calib.scale)               < 1e-9
    assert loaded.anatomical_zero == calib.anatomical_zero
