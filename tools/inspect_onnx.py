#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    args = ap.parse_args()

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    print("Inputs:")
    for i in sess.get_inputs():
        print(f"  name={i.name} shape={i.shape} type={i.type}")

    print("Outputs:")
    for o in sess.get_outputs():
        print(f"  name={o.name} shape={o.shape} type={o.type}")

if __name__ == "__main__":
    main()

