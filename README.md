# HER2 nanobody binder prediction

Two supervised tasks over a VHH (nanobody) library screened against HER2:

1. **Binder classification** — predict `binary_label` from sequence alone. Primary metric
   AUROC. Input restricted to frozen ESMC embeddings.
2. **Binding-strength regression** — predict `target_IE_score`. Primary metric Spearman ρ,
   with within-class ρ as the quantity of interest.

## Results (validation split)

| Task | Model | Metric | Reference |
| --- | --- | --- | --- |
| 1 | MLP on frozen ESMC layer 24 | **AUROC 0.9625 ± 0.0020** | 0.9430 (3-mer TF-IDF logreg) |
| 1 | Logistic regression, same embeddings | AUROC 0.9356 | — |
| 2 | Two-stage, seed ensemble | **ρ 0.6695**, within-class **+0.118 / +0.133** | ρ 0.6262, within +0.041 / +0.019 (3-mer ridge) |

Within-class ρ is the figure that matters in Task 2: overall ρ is dominated by the
binder/non-binder gap, so a pure classifier reaches ρ ≈ 0.63–0.84 with within-class ρ
indistinguishable from zero. [`docs/REPORT.pdf`](docs/REPORT.pdf) has the full analysis;
[`docs/mini_overview.pdf`](docs/mini_overview.pdf) places ten models on the nanobody
design pipeline.

## Three findings

**The labelling threshold is not what the task description states.** The documentation gives
`target_IE_score ≤ −1.32`, which agrees on 98.9% of rows. The exact rule is
`binary_label = 1 ⟺ target_IE_score < 0`.

**The signal in ESMC is present but not linearly accessible.** A linear probe on mean-pooled
embeddings (0.9356) underperforms 3-mer TF-IDF logistic regression (0.9430) on the same
split. One hidden layer reverses it (0.9625). This inverts the usual frozen-features
result, where the representation is decisive and the classifier incidental.

**CDR3-only pooling fails.** I predicted that pooling over CDR3 would beat whole-chain mean
pooling, on the reasoning that averaging over ~129 residues dilutes the ~15 that determine
binding. Measured, it is worse on both tasks and loses about a third of within-class ρ. The
dispersion statistic I had reasoned from measures variability, not discriminative utility.
See [`docs/REPORT.md`](docs/REPORT.md) §5.

## Layout

```
embed_esm.py          ESMC embedding: masked pooling, batch-invariance check, CDR3 pooling
train_task1.py        Task 1: logistic-regression baseline and MLP, layer sweep, grouped CV
train_task2.py        Task 2: two-stage model, target transforms, combination strategies
baselines_kmer.py     Sequence-only reference baselines (no GPU)
src/nanobody/data.py  Loading, split verification, the feature allowlist
src/nanobody/metrics.py  Metric bundles matching the graders, including within-class ρ
notebooks/            Colab notebook for the GPU embedding pass
tests/                Pooling correctness, CDR3 anchors, metric edge cases
docs/                 Report and pipeline overview (markdown and PDF)
results/              Predictions, per-model metrics, ablation tables
```

## Running it

```bash
uv venv --python 3.12
uv pip install -e .
uv run pytest tests/ -q
```

The embedding step needs the `embed` extra, which pulls the `esm` package:

```bash
uv run --extra embed python embed_esm.py --check                      # pooling correctness
uv run --extra embed python embed_esm.py --layers 6,12,18,24,30       # ~15 min on a T4
```

Then, on cached embeddings (CPU, seconds):

```bash
uv run python train_task1.py --sweep-layers
uv run python train_task1.py --layer 24 --seeds 0,1,2 --cv --wandb
uv run python train_task2.py --layer 24 --model two_stage --compare-combines --wandb
```

## Two implementation notes

**ESMC does not load via the `Auto*` classes.** `esm/__init__.py` exports nothing and never
imports `esm.models.esmc`, so the custom classes are never registered with transformers and
`tokenizer_class: EsmcTokenizer` cannot be resolved. `trust_remote_code=True` does not help,
because the class is in the installed package rather than the model repository.
`embed_esm.py` imports `EsmcTokenizer` and `EsmcForMaskedLM` directly.

**Pooling must exclude padding and special tokens.** The tokenizer prepends BOS, appends EOS,
and pads each batch to its longest member, so mean-pooling the raw hidden-state tensor makes
a sequence's vector depend on its batch-mates. `embed_esm.py --check` verifies the property
end to end; `tests/test_pooling.py` tests it on synthetic tensors, including a guard that
fails if pooling is replaced by a naive `.mean(dim=1)`.

## Data

The dataset is confidential and is **not** in this repository. `.gitignore` excludes
`data/`, the raw split files and the embedding caches. The scripts expect `../data/`
relative to the repository root.

`results/` holds the submitted predictions and the measured metrics, which are derived
outputs rather than source data.
