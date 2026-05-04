"""CLI entry point for Real2Sim (mink IK edition).

Modes
-----
--pose-only             webcam + skeleton overlay only (no MuJoCo)
--viewer                live MuJoCo viewer + camera preview window
--record OUT.mp4        write split-screen mp4 (camera | sim)
--duration N            stop after N seconds (default: run until 'q')
--camera IDX            webcam index (default 0)
--recalibrate           ignore cached calibration, run T-pose setup again
--ik-debug              print IK joint targets every 30 frames
--no-mirror             do not horizontally flip the camera display
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim.runner import RunOptions, run  # noqa: E402


def parse_args() -> RunOptions:
    p = argparse.ArgumentParser(description="Real2Sim — Unitree G1 webcam mirror (mink IK)")
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--pose-only",   action="store_true")
    p.add_argument("--viewer",      action="store_true")
    p.add_argument("--record",      type=str,   default=None, dest="record_path")
    p.add_argument("--duration",    type=float, default=0.0)
    p.add_argument("--no-mirror",   action="store_true")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--ik-debug",    action="store_true")
    a = p.parse_args()
    return RunOptions(
        camera        = a.camera,
        pose_only     = a.pose_only,
        viewer        = a.viewer,
        record_path   = Path(a.record_path) if a.record_path else None,
        duration      = a.duration,
        mirror_display= not a.no_mirror,
        debug_print   = a.debug,
        recalibrate   = a.recalibrate,
        ik_debug      = a.ik_debug,
    )


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
