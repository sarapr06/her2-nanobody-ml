"""Data loading, split integrity checks, and the feature allowlist.

The single most important thing in this module is `SEQUENCE_ONLY`: the rule that
nothing except the sequence may reach a model. See `column_verdicts()` for the
per-column reasoning that README section 2.3 asks for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column policy
# ---------------------------------------------------------------------------
# The test set ships with `id` and `sequence` only, so the operative question for
# every other column is: could this value exist for a nanobody that has never been
# through the screen? If not, using it is reading the answer off the assay.

TARGET_BINARY = "binary_label"
TARGET_REGRESSION = "target_IE_score"

#: The only column a model is allowed to consume as input.
SEQUENCE_ONLY = ("sequence",)


def column_verdicts() -> pd.DataFrame:
    """Per-column allow/deny verdict with the reason, for the report.

    `id`, `total`, `#Cluster` and `Cluster_size` are the cases worth arguing;
    everything else is mechanical.
    """
    rows = [
        # (column-or-group, verdict, reason)
        (
            "sequence",
            "ALLOW",
            "The input. Exists for any candidate, screened or not.",
        ),
        (
            "id",
            "DENY",
            "Assigned in descending order of total read count, so the integer in "
            "'ONU<n>' is a monotone function of library abundance -- an assay "
            "measurement wearing an identifier's clothes. A new clone has no id.",
        ),
        (
            "Marin-Her2* (15 cols)",
            "DENY",
            "Per-experiment enrichment against HER2. target_IE_score is an "
            "aggregation of exactly these columns (Spearman 0.92 vs their row "
            "mean), so they are not leakage-adjacent -- they are the answer.",
        ),
        (
            "Marin-NC* / Marin-Comparator* (11 cols)",
            "DENY",
            "Negative-control panning from the same sequencing run. Does not "
            "exist before the screen.",
        ),
        (
            "background_IE_score",
            "DENY",
            "Sibling aggregation over the control columns (Spearman 0.83 vs their "
            "row mean). Same sequencing run, same unavailability.",
        ),
        (
            "target_IE_score",
            "TARGET",
            "Task 2 target. Never a feature.",
        ),
        (
            "binary_label / label",
            "TARGET",
            "Task 1 target. It is a threshold on target_IE_score -- in this data "
            "exactly sign(target_IE_score) < 0 -- so feeding it to a Task 2 model "
            "hands over rho ~ 0.835 for free.",
        ),
        (
            "total",
            "DENY",
            "Total NGS read count for the clone, summed across all panning "
            "experiments including the HER2 ones. Abundance in the *output* of "
            "the screen is a consequence of binding, not a property of the "
            "molecule. An unscreened sequence has no read count.",
        ),
        (
            "Cluster",
            "DENY (as a feature); USE for grouping",
            "A 93%-identity cluster id. Computable for a new sequence in "
            "principle, but it is a nominal label with ~8.8k levels whose "
            "in-file label is 97.5% constant, so as a feature it is a "
            "near-lookup of the answer. It is however the correct CV grouping "
            "variable, and that is how this pipeline uses it.",
        ),
        (
            "#Cluster",
            "DENY",
            "Cluster index into the full library ordering (max 472,250 vs 8,763 "
            "clusters present here), i.e. derived from library-wide abundance "
            "like `id`. Not available pre-screen.",
        ),
        (
            "Cluster_size",
            "DENY -- the genuinely arguable one",
            "This is the hard case. It is NOT the number of rows this cluster "
            "contributes to the file (they match for only 1.4% of rows; median "
            "Cluster_size is 53 against a median of 1 row per cluster), so it "
            "counts cluster members in the *full unlabeled library*. That makes "
            "it a property of the alpaca's immune repertoire rather than of the "
            "HER2 panning: a large family means Marin expanded that lineage, "
            "which is weak evidence of antigen exposure and is arguably "
            "available for any sequence you can map back to the library. I still "
            "exclude it, for two reasons: (1) it cannot be computed for a "
            "genuinely novel design, which is the use case the whole exercise is "
            "about, and (2) clonal expansion in an immunized animal is itself "
            "partly a response to the immunogen, so it correlates with binding "
            "through a path that a fresh candidate does not have. I would report "
            "it as a covariate, never ship it as a feature.",
        ),
    ]
    return pd.DataFrame(rows, columns=["column", "verdict", "reason"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

CLIP_MAGNITUDE = 500.0


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Splits(train={len(self.train)}, val={len(self.val)}, "
            f"test={len(self.test)})"
        )


def load_splits(data_dir: str | Path = "../data") -> Splits:
    data_dir = Path(data_dir)
    return Splits(
        train=pd.read_csv(data_dir / "train.csv"),
        val=pd.read_csv(data_dir / "val.csv"),
        test=pd.read_csv(data_dir / "test_public.csv"),
    )


def verify_splits(splits: Splits) -> dict[str, object]:
    """Re-derive the claims README sections 2.2 and 2.4 make, rather than trusting them.

    Returns a dict of findings; `assert_ok` raises if a structural claim fails.
    """
    tr, va, te = splits.train, splits.val, splits.test
    both = pd.concat([tr, va], ignore_index=True)

    findings: dict[str, object] = {}

    # Sign convention. README says "more negative = more enriched"; check it on the
    # plain-HER2 columns rather than taking it on faith.
    plain_her2 = [
        c
        for c in tr.columns
        if "Her2" in c
        and not any(ab in c for ab in ("RabMAb", "Pertuzumab", "Trastuzumab"))
    ]
    for label in (1, 0):
        vals = tr.loc[tr[TARGET_BINARY] == label, plain_her2].to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        findings[f"plain_her2_frac_negative_label{label}"] = float((vals < 0).mean())
        findings[f"plain_her2_median_label{label}"] = float(np.median(vals))

    # The binary target is an exact sign threshold, not the approximate -1.32 the
    # README quotes (that rule agrees on only 98.9% of rows).
    findings["binary_is_exactly_sign"] = bool(
        ((both[TARGET_REGRESSION] < 0).astype(int) == both[TARGET_BINARY]).all()
    )
    findings["binary_vs_minus1_32_agreement"] = float(
        ((both[TARGET_REGRESSION] <= -1.32).astype(int) == both[TARGET_BINARY]).mean()
    )

    # Split integrity: cluster-disjoint, no repeated sequence anywhere.
    findings["cluster_overlap_train_val"] = len(set(tr.Cluster) & set(va.Cluster))
    findings["seq_overlap_train_val"] = len(set(tr.sequence) & set(va.sequence))
    findings["seq_overlap_train_test"] = len(set(tr.sequence) & set(te.sequence))
    findings["duplicate_seqs_within_train"] = int(tr.sequence.duplicated().sum())

    # Cluster structure -- why GroupKFold is mandatory.
    sizes = tr.groupby("Cluster")[TARGET_BINARY].agg(["mean", "size"])
    multi = sizes[sizes["size"] > 1]
    findings["n_clusters_train"] = int(len(sizes))
    findings["frac_singleton_clusters"] = float((sizes["size"] == 1).mean())
    findings["frac_rows_in_multi_clusters"] = float(multi["size"].sum() / len(tr))
    findings["frac_multi_clusters_label_constant"] = float(
        ((multi["mean"] == 0) | (multi["mean"] == 1)).mean()
    )

    # Cluster_size is not the in-file count -- the fact that drives its verdict.
    findings["cluster_size_matches_infile_count"] = float(
        (tr.groupby("Cluster").Cluster.transform("size") == tr.Cluster_size).mean()
    )

    findings["n_clipped_train"] = int((tr[TARGET_REGRESSION].abs() == CLIP_MAGNITUDE).sum())
    findings["n_clipped_val"] = int((va[TARGET_REGRESSION].abs() == CLIP_MAGNITUDE).sum())
    findings["pos_rate_train"] = float(tr[TARGET_BINARY].mean())
    findings["pos_rate_val"] = float(va[TARGET_BINARY].mean())
    return findings


def assert_ok(findings: dict[str, object]) -> None:
    """Fail loudly on the structural assumptions the pipeline depends on."""
    assert findings["cluster_overlap_train_val"] == 0, "splits are not cluster-disjoint"
    assert findings["seq_overlap_train_val"] == 0, "sequence leaks train->val"
    assert findings["seq_overlap_train_test"] == 0, "sequence leaks train->test"
    assert findings["duplicate_seqs_within_train"] == 0, "duplicate sequences in train"
    assert findings["binary_is_exactly_sign"], "binary_label is not sign(target)"


# ---------------------------------------------------------------------------
# Target handling
# ---------------------------------------------------------------------------


def drop_clipped(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the +/-500 'off the scale' markers.

    These are not measurements: 9 rows in train, 1 in val. The grader already
    excludes them from the test set. Keeping them in a regression training set
    would let ten rows dominate any squared-error gradient, and clipping them to
    the observed extreme would invent a number the assay never produced. We drop
    them for regression and keep them for classification, where only the sign is
    used and the sign is trustworthy.
    """
    return df[df[TARGET_REGRESSION].abs() != CLIP_MAGNITUDE].copy()


# ---------------------------------------------------------------------------
# Sequence anatomy (used for analysis and for the Task 2 CDR3-only ablation,
# never as a Task 1 feature -- Task 1 is ESMC embeddings only)
# ---------------------------------------------------------------------------

# CDR3 runs from just after the conserved cysteine at the end of framework 3 to the
# start of framework 4.
#
# Anchoring the right-hand end on 'WGQGT' loses 14% of the dataset, because that
# motif is only the most common variant: WGQGT 16007, WGKGT 1051, RGQGT 575,
# WGRGT 549, WGPGT 185, GGQGT 181. Every sequence here ends '...QVTVSSHHHHHH'
# though, so we anchor on QVTVSS and step back over the five-residue motif, which
# recovers 99.75% of sequences.
_FR4_ANCHOR = re.compile(r"QVTVSS")
_FR4_MOTIF_LEN = 5
_FR3_CYS = re.compile(r"Y[A-Z]C")

#: CDR3 lengths outside this range mean the anchors matched something unintended.
_CDR3_PLAUSIBLE_LEN = (5, 30)


def cdr3_span(sequence: str) -> tuple[int, int] | None:
    """Half-open character offsets [start, end) of CDR3, or None if unresolvable.

    Offsets rather than the substring, because `embed_esm.py --pool-cdr3` needs to
    map them onto residue positions in the token sequence. ESMC emits one token per
    residue, so the n-th real residue token corresponds to sequence character n.
    """
    anchor = _FR4_ANCHOR.search(sequence)
    if anchor is None:
        return None
    fr4_start = anchor.start() - _FR4_MOTIF_LEN
    if fr4_start <= 0:
        return None
    cys = list(_FR3_CYS.finditer(sequence, 0, fr4_start))
    if not cys:
        return None
    start = cys[-1].end()
    lo, hi = _CDR3_PLAUSIBLE_LEN
    return (start, fr4_start) if lo <= fr4_start - start <= hi else None


def extract_cdr3(sequence: str) -> str | None:
    """Return the CDR3 loop, or None if the framework anchors do not resolve.

    Used for analysis and for the CDR3 pooling ablation. Pooling ESMC hidden states
    over a different index set is still "ESMC embeddings only" under Task 1's rule --
    no k-mers, no one-hot, no hand-crafted descriptors enter the feature vector.
    """
    span = cdr3_span(sequence)
    return None if span is None else sequence[span[0] : span[1]]


def cdr3_coverage(sequences: pd.Series) -> float:
    return float(sequences.map(lambda s: extract_cdr3(s) is not None).mean())
