#!/usr/bin/env python
"""Task 1: binder classification from frozen ESMC embeddings.

    python train_task1.py --sweep-layers          # which layer to pool (val AUROC)
    python train_task1.py --layer 18              # train the chosen layer
    python train_task1.py --layer 18 --cv         # grouped CV on train, for honesty
    python train_task1.py --layer 18 --wandb      # log to wandb

Two models, both on the same mean-pooled ESMC vectors:

* `logreg`  - logistic regression, no hidden layer. The baseline the brief
              requires, and the number that says whether the MLP earns its keep.
* `mlp`     - Linear(960 -> h) -> ReLU -> Dropout -> Linear(h -> 1), BCEWithLogitsLoss.

Model selection happens on `val.csv`, which is cluster-disjoint from train. The
`--cv` path exists because a plain KFold on train would scatter near-duplicate
sequences across folds (83% of clusters are singletons but the other 17% hold 46%
of rows, with near-constant labels), so it reports GroupKFold-by-Cluster instead.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nanobody import data as D  # noqa: E402
from nanobody import metrics as M  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def load_embeddings(
    embed_dir: Path, split: str, layer: int, pooling: str = "whole"
) -> tuple[np.ndarray, np.ndarray]:
    """Load pooled vectors for one split.

    `pooling` selects the index set the ESMC hidden states were averaged over:

    * `whole`  - every real residue (the default; what Task 1's headline uses)
    * `cdr3`   - CDR3 residues only, testing whether whole-chain pooling dilutes
                 the signal that actually determines binding
    * `concat` - both, concatenated to 2*d_model

    All three are ESMC hidden states pooled differently, so all three stay inside
    Task 1's "ESMC embeddings only" constraint: no k-mers, no one-hot, no lengths,
    no hand-crafted descriptors enter the feature vector.
    """
    path = embed_dir / f"{split}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Run embed_esm.py first (see README.md); roughly "
            "15 minutes on a free Colab T4, ~50 minutes on Apple MPS."
        )
    with np.load(path, allow_pickle=False) as blob:
        def fetch(prefix: str) -> np.ndarray:
            key = f"{prefix}layer_{layer}"
            if key not in blob:
                available = sorted(k for k in blob.files if k.endswith(str(layer)))
                raise SystemExit(
                    f"{path} has no {key}; matching arrays are {available}. "
                    "For cdr3/concat pooling, re-run embed_esm.py --pool-cdr3."
                )
            return blob[key]

        if pooling == "whole":
            matrix = fetch("")
        elif pooling in ("cdr3", "concat"):
            whole, cdr3 = fetch(""), fetch("cdr3_").copy()
            # embed_esm.py stores NaN where the framework anchors did not resolve
            # (~0.4% of rows). Left alone, StandardScaler would spread those NaNs
            # across every column and the fit would fail or silently degrade.
            # Falling back to the whole-chain vector keeps the row and is the honest
            # default: "when CDR3 cannot be located, use the whole chain".
            unresolved = np.isnan(cdr3).any(axis=1)
            if unresolved.any():
                cdr3[unresolved] = whole[unresolved]
                print(
                    f"  {split}: {int(unresolved.sum())} rows "
                    f"({unresolved.mean():.2%}) had no locatable CDR3; "
                    "fell back to whole-chain pooling for those"
                )
            matrix = cdr3 if pooling == "cdr3" else np.concatenate([whole, cdr3], axis=1)
        else:
            raise SystemExit(f"unknown pooling {pooling!r}")
        if not np.isfinite(matrix).all():
            raise SystemExit(f"{path}: non-finite values remain after {pooling} pooling")
        return matrix, blob["ids"]


def aligned(
    frame: pd.DataFrame, embed_dir: Path, split: str, layer: int, pooling: str = "whole"
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return embeddings row-aligned with `frame`, matched on id.

    Never assume the cache and the CSV share a row order -- match explicitly, or a
    silent misalignment shows up as a mysteriously mediocre score.
    """
    matrix, ids = load_embeddings(embed_dir, split, layer, pooling)
    position = {str(sequence_id): i for i, sequence_id in enumerate(ids)}
    missing = [i for i in frame.id if str(i) not in position]
    if missing:
        raise SystemExit(
            f"{split}: {len(missing)} ids in the CSV are absent from the embedding "
            f"cache (first few: {missing[:5]}). Re-run embed_esm.py without --limit."
        )
    order = [position[str(i)] for i in frame.id]
    return matrix[order], frame


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_logreg(X_tr, y_tr, X_va, C: float = 1.0, seed: int = 0):
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    model.fit(scaler.transform(X_tr), y_tr)
    return model.predict_proba(scaler.transform(X_va))[:, 1], (scaler, model)


def fit_mlp(
    X_tr,
    y_tr,
    X_va,
    y_va,
    *,
    hidden: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    batch_size: int = 128,
    patience: int = 10,
    seed: int = 0,
    device: str = "cpu",
    run=None,
):
    """Train the MLP, early-stopping on val AUROC. Returns (val_scores, state)."""
    set_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    Xt = torch.tensor(scaler.transform(X_tr), dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(scaler.transform(X_va), dtype=torch.float32, device=device)

    model = MLP(Xt.shape[1], hidden, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(seed)
    best = {"auroc": -np.inf, "epoch": -1, "state": None, "scores": None}

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(Xt), generator=generator).to(device)
        epoch_loss = 0.0
        for start in range(0, len(Xt), batch_size):
            index = permutation[start : start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(Xt[index]), yt[index])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(index)
        epoch_loss /= len(Xt)

        model.eval()
        with torch.inference_mode():
            logits = model(Xv)
            val_loss = float(
                loss_fn(logits, torch.tensor(y_va, dtype=torch.float32, device=device))
            )
            scores = torch.sigmoid(logits).cpu().numpy()
        auroc = M.classification_metrics(y_va, scores)["auroc"]

        if run is not None:
            run.log(
                {"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss, "val_auroc": auroc}
            )
        if auroc > best["auroc"]:
            best = {
                "auroc": auroc,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "scores": scores,
            }
        elif epoch - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])
    return best["scores"], (scaler, model, best["epoch"])


def predict_mlp(state, X, device: str = "cpu") -> np.ndarray:
    scaler, model, _ = state
    model.eval()
    with torch.inference_mode():
        tensor = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
        return torch.sigmoid(model(tensor)).cpu().numpy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--embed-dir", type=Path, default=Path("../data/embeddings"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument(
        "--pooling",
        default="whole",
        choices=["whole", "cdr3", "concat"],
        help="which residues the ESMC hidden states were pooled over",
    )
    parser.add_argument("--sweep-layers", action="store_true", help="compare cached layers and exit")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated; results averaged")
    parser.add_argument("--cv", action="store_true", help="also run GroupKFold CV on train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)

    splits = D.load_splits(args.data_dir)
    D.assert_ok(D.verify_splits(splits))
    y_tr = splits.train.binary_label.to_numpy()
    y_va = splits.val.binary_label.to_numpy()

    # --- layer sweep -------------------------------------------------------
    if args.sweep_layers:
        with np.load(args.embed_dir / "train.npz", allow_pickle=False) as blob:
            layers = sorted(int(k.split("_")[1]) for k in blob.files if k.startswith("layer_"))
        rows = []
        for layer in layers:
            X_tr, _ = aligned(splits.train, args.embed_dir, "train", layer, args.pooling)
            X_va, _ = aligned(splits.val, args.embed_dir, "val", layer, args.pooling)
            scores, _ = fit_logreg(X_tr, y_tr, X_va, seed=seeds[0])
            auroc = M.classification_metrics(y_va, scores)["auroc"]
            rows.append({"layer": layer, "logreg_val_auroc": auroc})
            print(f"  layer {layer:2d}: logreg val AUROC = {auroc:.4f}")
        table = pd.DataFrame(rows)
        table.to_csv(args.outdir / "task1_layer_sweep.csv", index=False)
        best = table.loc[table.logreg_val_auroc.idxmax(), "layer"]
        print(f"\nbest layer by logreg val AUROC: {int(best)}")
        print("(swept with logistic regression -- cheap, and it ranks layers by how")
        print(" linearly separable the representation is, which is the property we want)")
        return

    # --- train both models -------------------------------------------------
    X_tr, _ = aligned(splits.train, args.embed_dir, "train", args.layer, args.pooling)
    X_va, _ = aligned(splits.val, args.embed_dir, "val", args.layer, args.pooling)
    X_te, _ = aligned(splits.test, args.embed_dir, "test_public", args.layer, args.pooling)
    print(
        f"layer {args.layer}, pooling={args.pooling}: "
        f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}"
    )

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="her2-nanobody",
            name=f"task1-layer{args.layer}",
            config=vars(args) | {"model": "mlp", "task": 1},
        )

    logreg_scores, _ = fit_logreg(X_tr, y_tr, X_va, seed=seeds[0])
    logreg_metrics = M.classification_metrics(y_va, logreg_scores)

    mlp_runs, test_predictions = [], []
    for seed in seeds:
        scores, state = fit_mlp(
            X_tr,
            y_tr,
            X_va,
            y_va,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
            run=run,
        )
        mlp_runs.append(M.classification_metrics(y_va, scores))
        test_predictions.append(predict_mlp(state, X_te, args.device))
        print(f"  seed {seed}: MLP val AUROC = {mlp_runs[-1]['auroc']:.4f}")

    mlp_mean = {k: float(np.mean([r[k] for r in mlp_runs])) for k in mlp_runs[0]}
    mlp_std = float(np.std([r["auroc"] for r in mlp_runs]))
    margin = mlp_mean["auroc"] - logreg_metrics["auroc"]

    print("\n=== Task 1, validation ===")
    print(f"  logistic regression : AUROC {logreg_metrics['auroc']:.4f}")
    print(f"  MLP (n={len(seeds)} seeds)   : AUROC {mlp_mean['auroc']:.4f} +/- {mlp_std:.4f}")
    print(f"  margin (MLP - logreg): {margin:+.4f}")
    if abs(margin) <= 2 * max(mlp_std, 1e-9):
        print("  -> inside seed noise: the embeddings did the work, not the classifier.")
    print("\n  reference AUROC (test set, from evaluate.py):")
    for name, value in M.TASK1_REFERENCES.items():
        print(f"    {value:.3f}  {name}")

    # --- grouped CV --------------------------------------------------------
    cv_summary = None
    if args.cv:
        groups = splits.train.Cluster.to_numpy()
        splitter = GroupKFold(n_splits=args.folds)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_tr, y_tr, groups)):
            scores, _ = fit_logreg(X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx], seed=seeds[0])
            fold_scores.append(M.classification_metrics(y_tr[va_idx], scores)["auroc"])
            print(f"  fold {fold}: logreg AUROC = {fold_scores[-1]:.4f}")
        cv_summary = {"mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores))}
        print(
            f"  GroupKFold(Cluster) logreg AUROC = {cv_summary['mean']:.4f} "
            f"+/- {cv_summary['std']:.4f}"
        )

    # --- submission --------------------------------------------------------
    # Average the per-seed test probabilities. Averaging probabilities (not ranks)
    # keeps the output inside [0, 1], which evaluate.py requires.
    test_score = np.mean(test_predictions, axis=0)
    submission = pd.DataFrame({"id": splits.test.id, "score": np.clip(test_score, 0.0, 1.0)})
    submission_path = args.outdir / "task1_predictions.csv"
    submission.to_csv(submission_path, index=False)

    # Val-format predictions, so evaluate.py can be run against a split we can see.
    pd.DataFrame({"id": splits.val.id, "score": np.mean([logreg_scores], axis=0)}).to_csv(
        args.outdir / "task1_val_logreg.csv", index=False
    )

    summary = {
        "layer": args.layer,
        "pooling": args.pooling,
        "seeds": seeds,
        "logreg": logreg_metrics,
        "mlp_mean": mlp_mean,
        "mlp_auroc_std": mlp_std,
        "margin_mlp_minus_logreg": margin,
        "grouped_cv_logreg": cv_summary,
        "hyperparameters": {
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    }
    (args.outdir / "task1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {submission_path} and {args.outdir / 'task1_summary.json'}")
    if run is not None:
        run.summary.update({"val_auroc_mlp": mlp_mean["auroc"], "val_auroc_logreg": logreg_metrics["auroc"]})
        run.finish()


if __name__ == "__main__":
    main()
