#!/usr/bin/env python
"""Sequence-only reference baselines, reproduced on our own val split.

    python baselines_kmer.py

The graders quote reference numbers measured on the withheld test set. We cannot
see that set, so this reproduces the same models on `val.csv` to establish what
the margin means on data we can actually inspect. Everything here uses the
sequence and nothing else.

Task 1 references (test-set AUROC, per evaluate.py):
    1-mer composition 0.799 | 1-NN 3-mer 0.931 | 3-mer logreg 0.958 | 3-mer TF-IDF 0.967
Task 2 references (test-set Spearman, per evaluate_regression.py):
    3-mer ridge raw 0.626 | 3-mer ridge rank 0.654 | class oracle 0.835

The Task 2 table also reports within-class rho, which is the number that matters:
the brief's claim is that the ridge model's 0.63 is entirely class separation.
This script checks that claim rather than repeating it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nanobody import data as D  # noqa: E402
from nanobody import metrics as M  # noqa: E402

SEED = 0


def char_tfidf(ngram: tuple[int, int] = (3, 3)) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char", ngram_range=ngram)


def run_task1(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    y_tr, y_va = train.binary_label.to_numpy(), val.binary_label.to_numpy()
    rows = []

    # Majority class: AUROC is 0.5 by construction, included as the floor.
    rows.append(("majority class", 0.5, None))

    # 1-mer composition: amino-acid frequencies only.
    vec = char_tfidf((1, 1))
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(vec.fit_transform(train.sequence), y_tr)
    scores = model.predict_proba(vec.transform(val.sequence))[:, 1]
    rows.append(("1-mer composition + logreg", M.classification_metrics(y_va, scores)["auroc"], scores))

    # 1-NN on 3-mer TF-IDF: no learning, pure nearest-neighbour retrieval.
    vec = char_tfidf()
    X_tr = vec.fit_transform(train.sequence)
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(X_tr, y_tr)
    scores = knn.predict_proba(vec.transform(val.sequence))[:, 1]
    rows.append(("1-NN 3-mer retrieval", M.classification_metrics(y_va, scores)["auroc"], scores))

    # 3-mer TF-IDF + logistic regression: the strongest reference, 0.967 on test.
    vec = char_tfidf()
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(vec.fit_transform(train.sequence), y_tr)
    scores = model.predict_proba(vec.transform(val.sequence))[:, 1]
    rows.append(("3-mer TF-IDF + logreg", M.classification_metrics(y_va, scores)["auroc"], scores))

    frame = pd.DataFrame(
        [(name, auroc) for name, auroc, _ in rows], columns=["model", "val_auroc"]
    )
    best = rows[-1][2]
    return frame, best


def run_task2(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Ridge on 3-mer TF-IDF, raw vs percentile-rank target, plus the class oracle."""
    # Clipped +/-500 rows are markers, not measurements: dropped from both sides so
    # our val number is comparable with the grader's test number.
    train = D.drop_clipped(train)
    val_scored = D.drop_clipped(val)

    y_tr = train.target_IE_score.to_numpy()
    y_va = val_scored.target_IE_score.to_numpy()
    cls_va = val_scored.binary_label.to_numpy()

    vec = char_tfidf()
    X_tr = vec.fit_transform(train.sequence)
    X_va = vec.transform(val_scored.sequence)

    rows = []

    # Raw target.
    ridge = Ridge(alpha=1.0, random_state=SEED).fit(X_tr, y_tr)
    rows.append(("3-mer ridge, raw target", M.regression_metrics(y_va, ridge.predict(X_va), cls_va)))

    # Percentile-rank target. Ranks are monotone in the raw target, so Spearman is
    # unaffected by the transform itself -- what changes is that the heavy left tail
    # stops dominating the squared-error gradient.
    ranks = pd.Series(y_tr).rank(pct=True).to_numpy()
    ridge_rank = Ridge(alpha=1.0, random_state=SEED).fit(X_tr, ranks)
    rows.append(
        ("3-mer ridge, percentile-rank target", M.regression_metrics(y_va, ridge_rank.predict(X_va), cls_va))
    )

    # Oracle that knows only the class and predicts that class's train median.
    medians = train.groupby("binary_label").target_IE_score.median()
    oracle = np.where(cls_va == 1, medians.loc[1], medians.loc[0]).astype(float)
    # Spearman needs a tie-break to be defined at all; with two distinct values it
    # is well-defined, and this reproduces the brief's 0.835 reference.
    rows.append(("oracle knowing only the class", M.regression_metrics(y_va, oracle, cls_va)))

    return pd.DataFrame(
        [
            {
                "model": name,
                "val_rho": m["spearman"],
                "within_binders": m["within_binders"],
                "within_non_binders": m["within_non_binders"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
            for name, m in rows
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    splits = D.load_splits(args.data_dir)
    findings = D.verify_splits(splits)
    D.assert_ok(findings)
    print("split integrity verified: cluster-disjoint, no repeated sequences\n")

    args.outdir.mkdir(parents=True, exist_ok=True)

    task1, _ = run_task1(splits.train, splits.val)
    print("=== Task 1 references (our val split; AUROC) ===")
    print(task1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    task1.to_csv(args.outdir / "baselines_task1.csv", index=False)

    task2 = run_task2(splits.train, splits.val)
    print("\n=== Task 2 references (our val split; Spearman) ===")
    print(task2.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    task2.to_csv(args.outdir / "baselines_task2.csv", index=False)

    print(f"\nwrote {args.outdir}/baselines_task1.csv and baselines_task2.csv")


if __name__ == "__main__":
    main()
