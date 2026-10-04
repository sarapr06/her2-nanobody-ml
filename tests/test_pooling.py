"""Correctness tests that do not need ESMC weights.

The pooling bug the brief warns about is entirely in `residue_mask` and
`masked_mean`, and both are pure tensor code. Testing them here means the
pooling is verified before spending GPU minutes, and `embed_esm.py --check`
then confirms the same property end to end on the real model.

    uv run pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from embed_esm import masked_mean, residue_mask  # noqa: E402
from nanobody import data as D  # noqa: E402
from nanobody import metrics as M  # noqa: E402


def test_masked_mean_matches_hand_computation():
    # Two sequences, d_model=2. Row 0 has 3 real residues, row 1 has 1.
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [999.0, 999.0]],
            [[7.0, 8.0], [111.0, 111.0], [222.0, 222.0], [333.0, 333.0]],
        ]
    )
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    expected = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    assert torch.allclose(masked_mean(hidden, mask), expected)


def test_residue_mask_drops_padding_and_special_tokens():
    encoded = {
        # BOS, residue, residue, EOS, PAD
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1]]),
    }
    assert residue_mask(encoded).tolist() == [[False, True, True, False, False]]


def test_masked_mean_is_invariant_to_padding_width():
    """The property that makes cached embeddings reproducible across batch sizes.

    Same sequence, two different amounts of right padding filled with garbage:
    the pooled vector must be identical.
    """
    torch.manual_seed(0)
    real = torch.randn(1, 5, 8)

    short = torch.cat([real, torch.full((1, 1, 8), 1e4)], dim=1)
    short_mask = torch.tensor([[True] * 5 + [False]])

    long = torch.cat([real, torch.full((1, 40, 8), -1e4)], dim=1)
    long_mask = torch.tensor([[True] * 5 + [False] * 40])

    assert torch.allclose(masked_mean(short, short_mask), masked_mean(long, long_mask))


def test_naive_mean_would_fail_the_same_check():
    """Guard against the test passing for the wrong reason.

    If someone replaces masked_mean with a plain `.mean(dim=1)`, the invariance
    test above must break -- otherwise it is not testing anything.
    """
    real = torch.ones(1, 5, 4)
    short = torch.cat([real, torch.zeros(1, 1, 4)], dim=1)
    long = torch.cat([real, torch.zeros(1, 40, 4)], dim=1)
    assert not torch.allclose(short.mean(dim=1), long.mean(dim=1))


# ---------------------------------------------------------------------------
# Sequence anatomy
# ---------------------------------------------------------------------------


# Synthetic VHH-like sequences, not library members. The dataset is confidential and
# must not appear in version control, so these are constructed to exercise the framework
# anchors (conserved Cys at the end of FR3, the FR4 motif, the terminal QVTVSSHHHHHH)
# without reproducing any real clone. `test_cdr3_coverage_on_the_real_data` below covers
# the shipped sequences, and skips when the data directory is absent.

def test_cdr3_extraction_with_canonical_framework4():
    sequence = (
        "MAQVQLQESGGGLVQAGGSLRLSCAASGSIFSINAMGWYRQAPGKQRELVATITSGGSTNYADSVKG"
        "RFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADRSGYWTLPGEYDYWGQGTQVTVSSHHHHHH"
    )
    assert D.extract_cdr3(sequence) == "AADRSGYWTLPGEYDY"


def test_cdr3_handles_non_canonical_framework4():
    # FR4 here is RGQGT, not WGQGT -- anchoring on WGQGT would lose this sequence, which
    # is why the extractor anchors on the universal terminal QVTVSS instead.
    sequence = (
        "MAQVQLQESGGGVVQPGGSLKLSCAASGFTFSSYWMYWVRQAPGKGLEWVSAINSDGSSTYYADSVKG"
        "RFTISRDNSKNTLYLQMNSLKPEDTALYYCAKGLYDSSGYAMDVRGQGTQVTVSSHHHHHH"
    )
    assert D.extract_cdr3(sequence) == "AKGLYDSSGYAMDV"


def test_cdr3_returns_none_without_anchors():
    assert D.extract_cdr3("MAQVQLQESGGG") is None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


#: Three rows per class -- `_safe_spearman` refuses fewer than three points,
#: since a rank correlation over two points is always +/-1 and meaningless.
_Y = np.array([-20.0, -10.0, -2.0, 3.0, 8.0, 40.0])
_CLS = np.array([1, 1, 1, 0, 0, 0])


def test_within_class_spearman_is_nan_for_a_class_constant_prediction():
    """The class oracle: informative overall, undefined within class.

    nan is the honest answer here, not 0.0 -- there is no ordering to score.
    """
    prediction = np.where(_CLS == 1, -5.0, 5.0)
    out = M.regression_metrics(_Y, prediction, _CLS)
    # Not 1.0: a two-valued prediction ties every member of a class together, and
    # Spearman penalises those ties. This is exactly why the real class oracle
    # tops out at rho ~ 0.835 rather than 1.0.
    assert 0.85 < out["spearman"] < 1.0
    assert np.isnan(out["within_binders"])
    assert np.isnan(out["within_non_binders"])
    assert np.isnan(out["within_mean"])


def test_within_class_spearman_detects_real_within_class_ordering():
    out = M.regression_metrics(_Y, _Y, _CLS)  # perfect predictions
    assert out["within_binders"] == pytest.approx(1.0)
    assert out["within_non_binders"] == pytest.approx(1.0)
    assert out["within_mean"] == pytest.approx(1.0)


def test_overall_spearman_can_be_high_while_within_class_is_zero():
    """The failure mode the Task 2 brief is built around.

    A pure classifier scores well overall and learns nothing inside a class. The
    metric bundle has to make that visible rather than hide it behind one number.
    """
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.uniform(-50, -2, 200), rng.uniform(2, 50, 200)])
    cls = np.concatenate([np.ones(200, int), np.zeros(200, int)])
    # Knows the class perfectly, orders randomly within it.
    prediction = np.where(cls == 1, -5.0, 5.0) + rng.normal(0, 0.01, 400)
    out = M.regression_metrics(y, prediction, cls)
    # ~0.75 here: the ceiling for a class-only predictor on two equal-sized groups.
    assert out["spearman"] > 0.7
    assert abs(out["within_binders"]) < 0.15
    assert abs(out["within_non_binders"]) < 0.15


DATA_DIR = ROOT.parent / "data"
needs_data = pytest.mark.skipif(
    not (DATA_DIR / "train.csv").exists(),
    reason="confidential dataset not present (expected in a fresh clone)",
)


@needs_data
def test_cdr3_coverage_on_the_real_data():
    """The anchors should resolve for almost every shipped sequence."""
    splits = D.load_splits(DATA_DIR)
    assert D.cdr3_coverage(splits.train.sequence) > 0.99


@needs_data
def test_sign_convention_holds_in_the_shipped_data():
    """More negative target == binder. Everything downstream depends on this."""
    splits = D.load_splits(DATA_DIR)
    findings = D.verify_splits(splits)
    D.assert_ok(findings)
    assert findings["binary_is_exactly_sign"]
    assert findings["plain_her2_frac_negative_label1"] > 0.9
    assert findings["plain_her2_frac_negative_label0"] < 0.01
