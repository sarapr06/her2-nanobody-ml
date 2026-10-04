#!/usr/bin/env python
"""Write fake embedding caches so the training scripts can be smoke-tested off-GPU.

    uv run python tests/make_synthetic_embeddings.py --outdir /tmp/fake_embeddings
    uv run python train_task1.py --embed-dir /tmp/fake_embeddings --layer 30 --epochs 3

This exists to check plumbing -- id alignment, shapes, early stopping, submission
format -- not modelling. The vectors carry a deliberately weak planted signal so
AUROC lands somewhere above chance and below the real thing; any score from these
files is meaningless as a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanobody import data as D  # noqa: E402

D_MODEL = 960
LAYERS = (6, 12, 18, 24, 30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/fake_embeddings"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    splits = D.load_splits(args.data_dir)

    # One shared direction that weakly encodes the label, so the classifier has
    # something learnable and id-alignment bugs surface as chance-level AUROC.
    direction = rng.normal(size=D_MODEL)
    direction /= np.linalg.norm(direction)

    for name, frame in (
        ("train", splits.train),
        ("val", splits.val),
        ("test_public", splits.test),
    ):
        n = len(frame)
        base = rng.normal(size=(n, D_MODEL)).astype(np.float32)
        if "binary_label" in frame.columns:
            shift = (frame.binary_label.to_numpy() * 2 - 1)[:, None] * direction[None, :]
            base += 0.35 * shift.astype(np.float32)
        arrays = {
            # Deeper layers get a slightly cleaner copy, so --sweep-layers has a
            # monotone trend to find rather than pure noise.
            f"layer_{layer}": (base + rng.normal(0, 0.6 - 0.015 * layer, base.shape)).astype(
                np.float32
            )
            for layer in LAYERS
        }
        arrays["ids"] = frame.id.to_numpy().astype("U")
        arrays["n_residues"] = frame.sequence.str.len().to_numpy()
        np.savez_compressed(args.outdir / f"{name}.npz", **arrays)
        print(f"{name}: {n} x {D_MODEL} -> {args.outdir / f'{name}.npz'}")

    print(f"\nSYNTHETIC DATA -- results from these files mean nothing.")


if __name__ == "__main__":
    main()
