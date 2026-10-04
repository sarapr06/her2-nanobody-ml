#!/usr/bin/env python
"""Task 2: binding-strength regression, judged on Spearman rho.

    python train_task2.py --transforms                 # transform comparison table
    python train_task2.py --model prob_map             # our own classifier baseline
    python train_task2.py --model two_stage            # the real model
    python train_task2.py --model two_stage --wandb

The brief's central point: overall rho is almost entirely the binder/non-binder
gap, so a pure classifier scores ~0.63-0.80 with within-class rho ~ 0.00. Every
model here therefore reports overall rho *and* both within-class rho values, and
model selection uses the within-class numbers.

Why the combination step is delicate
------------------------------------
A two-stage model has a class probability `p` and a within-class rank prediction.
The obvious combination, `pred = -W * p + w * r`, leaks: inside the binder class
`p` still varies, and if W >> w that variation dominates the ordering you just
worked to learn -- you get a good headline and within-class rho back near zero.

So the default combination predicts the *expected global percentile rank* instead.
Binders occupy the bottom pi of the global ordering and non-binders the top
(1 - pi), where pi = P(binder), so:

    rank = p * (pi * r_binder) + (1 - p) * (pi + (1 - pi) * r_non)

This is an expectation over the class posterior rather than an arbitrary weighting,
it is monotone in each input, and when `p` is confident the within-class term is
what orders members of that class. `--combine` exposes the alternatives so the
report can show what each choice costs.
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
import torch.nn.functional as F
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nanobody import data as D  # noqa: E402
from nanobody import metrics as M  # noqa: E402
from train_task1 import aligned, set_seed  # noqa: E402

TRANSFORMS = ("raw", "rank", "signed_log1p", "winsor")


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------


def transform_target(y: np.ndarray, kind: str, reference: np.ndarray | None = None):
    """Map the raw target to a training target.

    Spearman is invariant to any monotone transform of the *prediction*, so none of
    these can change the metric by themselves. What they change is which rows
    dominate the gradient: the raw target has skewness -10.2, so a squared-error
    loss spends almost everything on a handful of extreme clones.
    """
    reference = y if reference is None else reference
    if kind == "raw":
        return y.astype(np.float64)
    if kind == "rank":
        # Percentile rank within the training distribution.
        return pd.Series(y).rank(pct=True).to_numpy()
    if kind == "signed_log1p":
        return np.sign(y) * np.log1p(np.abs(y))
    if kind == "winsor":
        lo, hi = np.percentile(reference, [1, 99])
        return np.clip(y, lo, hi).astype(np.float64)
    raise ValueError(f"unknown transform {kind!r}")


def _ridge() -> RidgeCV:
    """Ridge with the penalty chosen by CV rather than left at alpha=1.

    On 960 standardised ESMC dimensions, `Ridge(alpha=1.0)` is ill-conditioned
    (scipy warns: rcond ~ 2e-08) because the pooled embedding dimensions are
    strongly collinear. The fit still returns, but the coefficients are numerically
    unreliable and not reproducible across BLAS versions. Selecting alpha over a
    wide grid moves the solution into a well-conditioned regime.

    The k-mer references in baselines_kmer.py deliberately keep `Ridge(alpha=1.0)`,
    since the brief specifies that exact configuration and those inputs are sparse
    TF-IDF, where it is not ill-conditioned.
    """
    return RidgeCV(alphas=np.logspace(-1, 4, 12))


def within_class_ranks(y: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Percentile rank of each row *within its own class*, ascending in y.

    0.0 is the most negative (strongest binder) member of the class, 1.0 the least.
    """
    out = np.zeros(len(y), dtype=np.float64)
    for label in (0, 1):
        mask = binary == label
        out[mask] = pd.Series(y[mask]).rank(pct=True).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TwoStageNet(nn.Module):
    """Shared trunk, three heads: class logit and one rank head per class.

    One trunk rather than two separate models, because the within-class heads have
    far less signal to learn from than the classifier and benefit from sharing the
    representation. The rank heads are sigmoid-bounded to [0, 1] to match the
    percentile-rank targets.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden, 1)
        self.rank_binder = nn.Linear(hidden, 1)
        self.rank_non = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (
            self.cls_head(h).squeeze(-1),
            torch.sigmoid(self.rank_binder(h)).squeeze(-1),
            torch.sigmoid(self.rank_non(h)).squeeze(-1),
        )


def pairwise_rank_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, margin: float = 0.05
) -> torch.Tensor:
    """Margin ranking loss over sampled within-class pairs.

    Spearman only cares about ordering, so a loss defined on pairs is a closer fit
    than MSE on values. Pairs are drawn only from rows in `mask` (one class), which
    is what keeps this from re-learning the class boundary.
    """
    index = torch.nonzero(mask, as_tuple=True)[0]
    if len(index) < 2:
        return predicted.sum() * 0.0
    permuted = index[torch.randperm(len(index), device=predicted.device)]
    a, b = index, permuted
    true_sign = torch.sign(target[a] - target[b])
    keep = true_sign != 0
    if keep.sum() == 0:
        return predicted.sum() * 0.0
    return F.margin_ranking_loss(
        predicted[a][keep], predicted[b][keep], true_sign[keep], margin=margin
    )


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def combine(
    p: np.ndarray,
    r_binder: np.ndarray,
    r_non: np.ndarray,
    pi: float,
    how: str,
    weight: float = 1.0,
) -> np.ndarray:
    """Turn (class probability, within-class ranks) into one number to be ranked.

    Returns something monotone in "predicted score", i.e. small = strong binder,
    matching the target's sign convention.
    """
    if how == "expected_rank":
        # Expectation over the class posterior; see the module docstring.
        return p * (pi * r_binder) + (1.0 - p) * (pi + (1.0 - pi) * r_non)
    if how == "weighted":
        # The naive version, kept so the report can quantify what it costs.
        return -100.0 * p + weight * np.where(p >= 0.5, r_binder, r_non)
    if how == "hard":
        # Hard class assignment: protects within-class ordering completely, but a
        # misclassified clone is placed at the wrong end of the whole ordering.
        predicted_class = (p >= 0.5).astype(float)
        return -predicted_class + weight * np.where(
            predicted_class == 1, r_binder * pi, r_non * (1 - pi)
        )
    raise ValueError(f"unknown combine mode {how!r}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_two_stage(
    X_tr,
    y_tr,
    cls_tr,
    X_va,
    *,
    hidden=256,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=80,
    batch_size=256,
    patience=15,
    rank_weight=1.0,
    pair_weight=1.0,
    seed=0,
    device="cpu",
    val_eval=None,
    run=None,
):
    """Train the joint model. Early-stops on mean within-class rho, not overall rho.

    Selecting on overall rho would pick the best classifier; the brief is explicit
    that the within-class ordering is the part worth having.
    """
    set_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    Xt = torch.tensor(scaler.transform(X_tr), dtype=torch.float32, device=device)
    Xv = torch.tensor(scaler.transform(X_va), dtype=torch.float32, device=device)

    ranks = within_class_ranks(y_tr, cls_tr)
    rt = torch.tensor(ranks, dtype=torch.float32, device=device)
    ct = torch.tensor(cls_tr, dtype=torch.float32, device=device)

    model = TwoStageNet(Xt.shape[1], hidden, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    best = {"score": -np.inf, "epoch": -1, "state": None}

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(Xt), generator=generator).to(device)
        totals = {"loss": 0.0, "cls": 0.0, "rank": 0.0, "pair": 0.0}

        for start in range(0, len(Xt), batch_size):
            index = permutation[start : start + batch_size]
            xb, cb, rb = Xt[index], ct[index], rt[index]
            is_binder, is_non = cb == 1, cb == 0

            optimiser.zero_grad()
            logit, r_bind, r_non = model(xb)

            loss_cls = bce(logit, cb)
            # Each rank head is supervised only on its own class's rows.
            loss_rank = xb.new_zeros(())
            for mask, prediction in ((is_binder, r_bind), (is_non, r_non)):
                if mask.any():
                    loss_rank = loss_rank + F.mse_loss(prediction[mask], rb[mask])
            loss_pair = xb.new_zeros(())
            if pair_weight > 0:
                for mask, prediction in ((is_binder, r_bind), (is_non, r_non)):
                    loss_pair = loss_pair + pairwise_rank_loss(prediction, rb, mask)

            loss = loss_cls + rank_weight * loss_rank + pair_weight * loss_pair
            loss.backward()
            optimiser.step()

            scale = len(index)
            totals["loss"] += float(loss.detach()) * scale
            totals["cls"] += float(loss_cls.detach()) * scale
            totals["rank"] += float(loss_rank.detach()) * scale
            totals["pair"] += float(loss_pair.detach()) * scale

        totals = {k: v / len(Xt) for k, v in totals.items()}

        model.eval()
        with torch.inference_mode():
            logit, r_bind, r_non = model(Xv)
            outputs = (
                torch.sigmoid(logit).cpu().numpy(),
                r_bind.cpu().numpy(),
                r_non.cpu().numpy(),
            )

        logged = {"epoch": epoch, **{f"train_{k}": v for k, v in totals.items()}}
        score = None
        if val_eval is not None:
            val_metrics = val_eval(*outputs)
            score = val_metrics["within_mean"]
            logged.update(
                {
                    "val_spearman": val_metrics["spearman"],
                    "val_within_binders": val_metrics["within_binders"],
                    "val_within_non_binders": val_metrics["within_non_binders"],
                    "val_within_mean": score,
                }
            )
        if run is not None:
            run.log(logged)

        if score is not None and np.isfinite(score) and score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, scaler, best["epoch"]


@torch.inference_mode()
def predict_heads(model, scaler, X, device="cpu"):
    model.eval()
    tensor = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    logit, r_bind, r_non = model(tensor)
    return (
        torch.sigmoid(logit).cpu().numpy(),
        r_bind.cpu().numpy(),
        r_non.cpu().numpy(),
    )


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
        "--pooling", default="whole", choices=["whole", "cdr3", "concat"]
    )
    parser.add_argument(
        "--model", default="two_stage", choices=["two_stage", "prob_map", "single_head"]
    )
    parser.add_argument("--transform", default="rank", choices=TRANSFORMS)
    parser.add_argument(
        "--transforms", action="store_true", help="run the transform comparison and exit"
    )
    parser.add_argument(
        "--combine", default="expected_rank", choices=["expected_rank", "weighted", "hard"]
    )
    parser.add_argument("--compare-combines", action="store_true")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)

    splits = D.load_splits(args.data_dir)
    D.assert_ok(D.verify_splits(splits))

    # The +/-500 markers are dropped from training and from our val metrics, since
    # the grader drops them from the test set. See data.drop_clipped.
    train = D.drop_clipped(splits.train)
    val = D.drop_clipped(splits.val)
    print(
        f"dropped {len(splits.train) - len(train)} clipped train rows, "
        f"{len(splits.val) - len(val)} clipped val rows"
    )

    X_tr, _ = aligned(train, args.embed_dir, "train", args.layer, args.pooling)
    X_va, _ = aligned(val, args.embed_dir, "val", args.layer, args.pooling)
    X_te, _ = aligned(splits.test, args.embed_dir, "test_public", args.layer, args.pooling)

    y_tr, y_va = train.target_IE_score.to_numpy(), val.target_IE_score.to_numpy()
    cls_tr, cls_va = train.binary_label.to_numpy(), val.binary_label.to_numpy()
    pi = float(cls_tr.mean())
    print(f"layer {args.layer}: train {X_tr.shape}, val {X_va.shape}; P(binder)={pi:.3f}")

    def evaluate(predictions) -> dict[str, float]:
        return M.regression_metrics(y_va, predictions, cls_va)

    # --- transform comparison (ridge, so it is fast and about the transform) ---
    if args.transforms:
        rows = []
        for kind in TRANSFORMS:
            target = transform_target(y_tr, kind)
            scaler = StandardScaler().fit(X_tr)
            model = _ridge().fit(scaler.transform(X_tr), target)
            metrics = evaluate(model.predict(scaler.transform(X_va)))
            rows.append({"transform": kind, **{k: metrics[k] for k in
                        ("spearman", "within_binders", "within_non_binders", "mae", "r2")}})
            print(f"  {kind:14s} {M.format_regression(metrics)}")
        table = pd.DataFrame(rows)
        table.to_csv(args.outdir / "task2_transforms.csv", index=False)
        print(f"\nwrote {args.outdir / 'task2_transforms.csv'}")
        print("Note: Spearman is invariant to monotone transforms of the prediction,")
        print("so any difference here comes from which rows dominate the loss.")
        return

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="her2-nanobody",
            name=f"task2-{args.model}-layer{args.layer}",
            config=vars(args) | {"task": 2},
        )

    results: dict[str, dict] = {}
    test_prediction = None

    # --- our own baseline: a monotone map from the Task 1 probability ----------
    if args.model == "prob_map":
        # Predict the class median, ordered by classifier probability. This is the
        # "classifier in a trench coat" the brief warns about -- built deliberately
        # so its within-class rho can be shown to be ~0.
        from train_task1 import fit_logreg

        p_va, _ = fit_logreg(X_tr, cls_tr, X_va, seed=seeds[0])
        medians = pd.Series(y_tr).groupby(cls_tr).median()
        prediction = np.where(p_va >= 0.5, medians.loc[1], medians.loc[0]) - p_va
        metrics = evaluate(prediction)
        val_prediction = prediction
        results["prob_map"] = metrics
        print(f"\nprob_map (our classifier baseline): {M.format_regression(metrics)}")
        p_te, _ = fit_logreg(X_tr, cls_tr, X_te, seed=seeds[0])
        test_prediction = np.where(p_te >= 0.5, medians.loc[1], medians.loc[0]) - p_te

    # --- single-head regression on a transformed target ----------------------
    elif args.model == "single_head":
        scaler = StandardScaler().fit(X_tr)
        target = transform_target(y_tr, args.transform)
        model = _ridge().fit(scaler.transform(X_tr), target)
        val_prediction = model.predict(scaler.transform(X_va))
        metrics = evaluate(val_prediction)
        results[f"single_head_{args.transform}"] = metrics
        print(f"\nsingle_head ({args.transform}): {M.format_regression(metrics)}")
        test_prediction = model.predict(scaler.transform(X_te))

    # --- the real model ------------------------------------------------------
    else:
        per_seed, test_parts, val_parts = [], [], []
        for seed in seeds:
            model, scaler, best_epoch = fit_two_stage(
                X_tr,
                y_tr,
                cls_tr,
                X_va,
                hidden=args.hidden,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                rank_weight=args.rank_weight,
                pair_weight=args.pair_weight,
                seed=seed,
                device=args.device,
                val_eval=lambda p, rb, rn: evaluate(
                    combine(p, rb, rn, pi, args.combine)
                ),
                run=run,
            )
            heads_va = predict_heads(model, scaler, X_va, args.device)
            metrics = evaluate(combine(*heads_va, pi, args.combine))
            per_seed.append(metrics)
            val_parts.append(heads_va)
            test_parts.append(predict_heads(model, scaler, X_te, args.device))
            print(
                f"  seed {seed} (best epoch {best_epoch}): {M.format_regression(metrics)}"
            )

            if args.compare_combines and seed == seeds[0]:
                print("\n  combination comparison (same trained model):")
                rows = []
                for how in ("expected_rank", "weighted", "hard"):
                    alt = evaluate(combine(*heads_va, pi, how))
                    rows.append({"combine": how, **{k: alt[k] for k in
                                ("spearman", "within_binders", "within_non_binders")}})
                    print(f"    {how:14s} {M.format_regression(alt)}")
                pd.DataFrame(rows).to_csv(args.outdir / "task2_combine.csv", index=False)

        mean = {
            k: float(np.nanmean([m[k] for m in per_seed]))
            for k in per_seed[0]
            if isinstance(per_seed[0][k], float)
        }
        std = float(np.nanstd([m["spearman"] for m in per_seed]))
        results["two_stage"] = mean | {"spearman_std": std}
        print(f"\ntwo_stage mean over {len(seeds)} seeds:")
        print(f"  {M.format_regression(mean)}  (rho sd {std:.4f})")

        # Average the class posterior and each rank head across seeds, then combine
        # once. Combining per seed and averaging the results would mix ranks from
        # different orderings, which is not a meaningful average.
        def averaged(parts):
            return combine(*[np.mean([h[i] for h in parts], axis=0) for i in range(3)],
                           pi, args.combine)

        test_prediction = averaged(test_parts)
        val_prediction = averaged(val_parts)
        # The seed-averaged model is what actually ships, so report its val metrics
        # rather than only the per-seed mean above.
        ensemble = evaluate(val_prediction)
        results["two_stage_seed_ensemble"] = ensemble
        print(f"  seed-averaged ensemble: {M.format_regression(ensemble)}")

    print("\n  reference Spearman (test set, from evaluate_regression.py):")
    for name, value in M.TASK2_REFERENCES.items():
        print(f"    {value:.3f}  {name}")

    submission = pd.DataFrame({"id": splits.test.id, "prediction": test_prediction})
    path = args.outdir / "task2_predictions.csv"
    submission.to_csv(path, index=False)

    # Same format against a split we have labels for, so the real grader can be run
    # before submitting:
    #   python ../evaluate_regression.py results/task2_val_predictions.csv \
    #       --truth ../data/val.csv
    # The clipped val row is excluded above, so it is re-added with the class median
    # to keep every val id present; the grader drops it from the metrics anyway.
    val_predictions = pd.DataFrame({"id": val.id, "prediction": val_prediction})
    missing_ids = set(splits.val.id) - set(val_predictions.id)
    if missing_ids:
        filler = float(np.median(val_prediction))
        val_predictions = pd.concat(
            [val_predictions, pd.DataFrame({"id": sorted(missing_ids), "prediction": filler})],
            ignore_index=True,
        )
    val_predictions.to_csv(args.outdir / "task2_val_predictions.csv", index=False)
    (args.outdir / "task2_summary.json").write_text(
        json.dumps({"model": args.model, "combine": args.combine, "layer": args.layer,
                    "pooling": args.pooling,
                    "pi": pi, "seeds": seeds, "results": results}, indent=2, default=float)
    )
    print(f"\nwrote {path} and {args.outdir / 'task2_summary.json'}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
