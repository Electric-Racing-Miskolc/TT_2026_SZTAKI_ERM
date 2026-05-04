"""Enumerate camera devices on Windows and report which give real frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def try_index(idx: int, backend: int, backend_name: str) -> dict:
    cap = cv2.VideoCapture(idx, backend)
    info = {"idx": idx, "backend": backend_name, "opened": False, "frame": None}
    if not cap.isOpened():
        return info
    info["opened"] = True
    info["w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Some cameras return junk on the first read; grab a few.
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            info["frame"] = frame
            break
    cap.release()
    return info


def describe(info: dict) -> str:
    if not info["opened"]:
        return f"  idx={info['idx']:>2} {info['backend']:<10} -> not opened"
    if info.get("frame") is None:
        return f"  idx={info['idx']:>2} {info['backend']:<10} -> opened {info['w']}x{info['h']} but no frame"
    f = info["frame"]
    mean = float(np.mean(f))
    nonzero = int(np.count_nonzero(f.sum(axis=-1)))
    total = f.shape[0] * f.shape[1]
    return (f"  idx={info['idx']:>2} {info['backend']:<10} -> {info['w']}x{info['h']} "
            f"mean={mean:5.1f} nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)")


def main() -> int:
    out_dir = Path(__file__).parent / "camera_diag"
    out_dir.mkdir(exist_ok=True)

    print("Camera enumeration (this may produce harmless warnings):")
    findings = []
    for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
        for idx in range(4):
            info = try_index(idx, backend, name)
            findings.append(info)
            print(describe(info))
            if info.get("frame") is not None:
                p = out_dir / f"cam_{name}_{idx}.jpg"
                cv2.imwrite(str(p), info["frame"])

    candidates = [f for f in findings if f.get("frame") is not None]
    print(f"\n{len(candidates)} working camera index/backend combos.")
    for f in candidates:
        print(f"  -> {f['backend']} idx {f['idx']}: {f['w']}x{f['h']}")
    print(f"frames saved under {out_dir}")
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
