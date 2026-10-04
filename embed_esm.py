#!/usr/bin/env python
"""Extract frozen ESMC embeddings for every sequence in train/val/test.

    python embed_esm.py --limit 32     # ~1 minute, confirms the install works
    python embed_esm.py --check        # batch-invariance test, no cache written
    python embed_esm.py                # all 19,367 sequences -> outdir/*.npz

ESMC stays frozen; this script only ever runs it under `inference_mode`.

Two details decide whether the output is usable:

1. **Pooling.** The tokenizer prepends BOS, appends EOS, and pads every sequence to
   the longest in its batch. Mean-pooling the raw hidden-state tensor therefore
   averages in padding and special tokens, which makes a sequence's vector depend
   on which other sequences shared its batch. We build the mask from
   `attention_mask & ~special_tokens_mask` so only real residues contribute.
   `--check` verifies this empirically.

2. **Layer choice.** The last layer of a masked-LM is specialised for token
   reconstruction and is often not the best sentence-level representation, so we
   cache several layers in one pass and let Task 1 pick on validation. Pooled
   vectors are cheap (~74 MB per layer for 19k sequences at d_model=960); the
   per-residue tensors are not (~9.5 GB for one layer), so those are opt-in via
   `--save-per-residue` for the attention/CNN variant.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nanobody.data import cdr3_span  # noqa: E402

MODEL_ID = "biohub/ESMC-300M"
N_LAYERS = 30  # ESMC-300M: 30 transformer layers, d_model 960
DEFAULT_LAYERS = (6, 12, 18, 24, 30)
SPLITS = ("train", "val", "test_public")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device):
    """Load ESMC, frozen.

    The task brief suggests `AutoTokenizer`/`AutoModelForMaskedLM` after a bare
    `import esm`, but that fails with:

        ValueError: Tokenizer class EsmcTokenizer does not exist or is not
        currently imported.

    `esm/__init__.py` exports nothing and does not import `esm.models.esmc`, so
    the custom classes are never registered with transformers and the Auto
    factories cannot resolve the `tokenizer_class: EsmcTokenizer` declared in the
    repo's tokenizer_config.json. `trust_remote_code=True` does not help either --
    the class lives in the installed package, not in the model repo.

    So we import the concrete classes and skip Auto resolution entirely. Fewer
    moving parts, and it does not depend on import side effects.
    """
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

    tokenizer = EsmcTokenizer.from_pretrained(MODEL_ID)
    model = EsmcForMaskedLM.from_pretrained(MODEL_ID).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def residue_mask(encoded) -> torch.Tensor:
    """True exactly at real residues: padding and BOS/EOS removed."""
    mask = encoded["attention_mask"].bool()
    special = encoded.get("special_tokens_mask")
    if special is not None:
        mask = mask & ~special.bool()
    return mask


def span_mask(
    mask: torch.Tensor, spans: list[tuple[int, int] | None]
) -> torch.Tensor:
    """Narrow a residue mask to a per-sequence character span.

    ESMC emits one token per residue, so the n-th True position in `mask` (in order)
    is sequence character n. That lets a character span from `data.cdr3_span` select
    token positions without re-deriving the tokenizer's offsets.

    Rows whose span is None keep an all-False mask; callers must treat their pooled
    vector as missing rather than as a zero vector.
    """
    out = torch.zeros_like(mask)
    for row, span in enumerate(spans):
        if span is None:
            continue
        positions = torch.nonzero(mask[row], as_tuple=True)[0]
        start, end = span
        selected = positions[start:end]
        if len(selected):
            out[row, selected] = True
    return out


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real residues only.

    hidden: (batch, length, d_model); mask: (batch, length).
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return summed / counts


@torch.inference_mode()
def embed_batch(
    model,
    tokenizer,
    sequences: list[str],
    layers: tuple[int, ...],
    device: torch.device,
    per_residue_layer: int | None = None,
    cdr_spans: list[tuple[int, int] | None] | None = None,
) -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    list[np.ndarray] | None,
    dict[int, np.ndarray] | None,
]:
    """Pooled vectors per requested layer, residue counts, and optional per-residue.

    The per-residue return is a ragged list of (n_residues, d_model) arrays with
    padding and special tokens already stripped, so downstream code never has to
    re-derive the mask.
    """
    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        return_special_tokens_mask=True,
    )
    special = encoded.pop("special_tokens_mask")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output = model(**encoded, output_hidden_states=True)

    mask = residue_mask({**encoded, "special_tokens_mask": special.to(device)})
    # hidden_states is (embeddings, layer_1, ..., layer_N), so index i == layer i.
    pooled = {
        layer: masked_mean(output.hidden_states[layer], mask).float().cpu().numpy()
        for layer in layers
    }
    counts = mask.sum(dim=1).cpu().numpy()

    residues = None
    if per_residue_layer is not None:
        hidden = output.hidden_states[per_residue_layer].float().cpu().numpy()
        mask_cpu = mask.cpu().numpy()
        residues = [row[keep] for row, keep in zip(hidden, mask_cpu)]

    # CDR3-only pooling, computed here on the GPU so the cache stays the same size as
    # the whole-chain pooled vectors. Caching per-residue tensors instead would mean
    # ~4.8 GB for 19k sequences, for the sake of one weighted average.
    cdr_pooled = None
    if cdr_spans is not None:
        narrowed = span_mask(mask, cdr_spans)
        cdr_pooled = {
            layer: masked_mean(output.hidden_states[layer], narrowed)
            .float()
            .cpu()
            .numpy()
            for layer in layers
        }
        # Mark unresolvable rows so they are never mistaken for a real zero vector.
        empty = (~narrowed.any(dim=1)).cpu().numpy()
        for layer in layers:
            cdr_pooled[layer][empty] = np.nan

    return pooled, counts, residues, cdr_pooled


def batch_invariance_check(
    model, tokenizer, sequences: list[str], device: torch.device, layer: int = N_LAYERS
) -> None:
    """Embed one sequence alone and inside a mixed-length batch; compare.

    If pooling is wrong the two differ well outside floating-point tolerance,
    because the batched version averages in padding.
    """
    target = sequences[0]
    lone, _, _, _ = embed_batch(model, tokenizer, [target], (layer,), device)
    # Sort by length so `target` sits with very differently-sized neighbours and
    # the batch is padded far beyond its own length.
    mixed = [target] + sorted(sequences[1:32], key=len, reverse=True)
    batched, _, _, _ = embed_batch(model, tokenizer, mixed, (layer,), device)

    a, b = lone[layer][0], batched[layer][0]
    max_abs = float(np.abs(a - b).max())
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    pad_ratio = max(len(s) for s in mixed) / len(target)

    print(f"batch-invariance check  (layer {layer}, alone vs batch of {len(mixed)})")
    print(f"  longest/target length ratio = {pad_ratio:.2f}x")
    print(f"  max |difference|            = {max_abs:.3e}")
    print(f"  cosine similarity           = {cosine:.8f}")
    tolerance = 1e-3  # fp32 accumulation over a padded batch is not bit-exact
    if max_abs < tolerance:
        print("  PASS - pooling ignores padding and special tokens")
    else:
        raise SystemExit(
            f"  FAIL - max difference {max_abs:.3e} exceeds {tolerance:.0e}; "
            "pooling is averaging in padding or special tokens"
        )


def load_sequences(data_dir: Path, split: str, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{split}.csv", usecols=["id", "sequence"])
    return frame.head(limit) if limit else frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--outdir", type=Path, default=Path("../data/embeddings"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N rows per split")
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(str(n) for n in DEFAULT_LAYERS),
        help=f"comma-separated hidden-state indices, 1-{N_LAYERS}",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the batch-invariance test and exit without writing a cache",
    )
    parser.add_argument(
        "--pool-cdr3",
        action="store_true",
        help="additionally pool over CDR3 residues only (tests the dilution hypothesis)",
    )
    parser.add_argument(
        "--save-per-residue",
        action="store_true",
        help="also cache the last-layer per-residue tensors (large; for the attention model)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits",
        default=",".join(SPLITS),
        help=(
            "comma-separated subset of splits to embed. Existing outputs are skipped "
            "anyway, so this is for running one split at a time under a time limit."
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    layers = tuple(int(x) for x in args.layers.split(","))

    device = pick_device(args.device)
    print(f"device={device}  model={MODEL_ID}  layers={layers}")
    model, tokenizer = load_model(device)

    # Validate against the model we actually loaded rather than the layer count in
    # the brief: hidden_states has num_hidden_layers + 1 entries (index 0 is the
    # embedding output), so a silently out-of-range index would either crash deep in
    # the loop or, worse, wrap around and cache the wrong layer.
    depth = getattr(model.config, "num_hidden_layers", N_LAYERS)
    width = getattr(model.config, "hidden_size", None)
    print(f"model depth={depth} layers, d_model={width}")
    if depth != N_LAYERS:
        print(f"note: expected {N_LAYERS} layers from the brief, model reports {depth}")
    if not all(1 <= layer <= depth for layer in layers):
        raise SystemExit(f"--layers must be within 1-{depth} for this model, got {layers}")

    if args.check:
        sequences = load_sequences(args.data_dir, "train", 64).sequence.tolist()
        batch_invariance_check(model, tokenizer, sequences, device, layer=depth)
        return

    requested_splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    unknown = set(requested_splits) - set(SPLITS)
    if unknown:
        raise SystemExit(f"unknown split(s) {sorted(unknown)}; choose from {list(SPLITS)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for split in requested_splits:
        destination = args.outdir / f"{split}.npz"
        if destination.exists() and not args.overwrite and not args.limit:
            print(f"{split}: {destination} exists, skipping (use --overwrite)")
            continue

        frame = load_sequences(args.data_dir, split, args.limit)
        pooled: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        lengths: list[np.ndarray] = []
        per_residue: list[np.ndarray] = []
        started = time.time()

        residue_layer = max(layers) if args.save_per_residue else None
        cdr_pooled_acc: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        if args.pool_cdr3:
            resolved = frame.sequence.map(lambda q: cdr3_span(q) is not None).mean()
            print(f"  {split}: CDR3 anchors resolve for {resolved:.4%} of sequences")
        for start in range(0, len(frame), args.batch_size):
            chunk = frame.sequence.iloc[start : start + args.batch_size].tolist()
            spans = [cdr3_span(q) for q in chunk] if args.pool_cdr3 else None
            batch_pooled, batch_lengths, batch_residues, batch_cdr = embed_batch(
                model,
                tokenizer,
                chunk,
                layers,
                device,
                per_residue_layer=residue_layer,
                cdr_spans=spans,
            )
            if batch_cdr is not None:
                for layer in layers:
                    cdr_pooled_acc[layer].append(batch_cdr[layer])
            for layer in layers:
                pooled[layer].append(batch_pooled[layer])
            lengths.append(batch_lengths)
            if batch_residues is not None:
                per_residue.extend(batch_residues)
            done = min(start + args.batch_size, len(frame))
            if start % (args.batch_size * 20) == 0 or done == len(frame):
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {split}: {done}/{len(frame)}  ({rate:.1f} seq/s)", flush=True)

        arrays = {f"layer_{layer}": np.concatenate(pooled[layer]) for layer in layers}
        # Fixed-width unicode, not object dtype: object arrays would force
        # allow_pickle=True on every read of the cache.
        if args.pool_cdr3:
            arrays.update(
                {
                    f"cdr3_layer_{layer}": np.concatenate(cdr_pooled_acc[layer])
                    for layer in layers
                }
            )
        arrays["ids"] = frame.id.to_numpy().astype("U")
        arrays["n_residues"] = np.concatenate(lengths)
        np.savez_compressed(destination, **arrays)

        if per_residue:
            # Ragged tensors stored flat with offsets: sequence i occupies
            # rows offsets[i]:offsets[i + 1]. Avoids pickled object arrays.
            offsets = np.cumsum([0] + [len(r) for r in per_residue])
            residue_path = args.outdir / f"{split}_per_residue_layer{max(layers)}.npz"
            np.savez(
                residue_path,
                residues=np.concatenate(per_residue).astype(np.float16),
                offsets=offsets,
                ids=frame.id.to_numpy().astype("U"),
            )
            print(
                f"{split}: wrote {residue_path} "
                f"({residue_path.stat().st_size / 1e6:.0f} MB, float16)"
            )
        elapsed = time.time() - started
        print(
            f"{split}: wrote {destination} "
            f"({len(frame)} seqs, {elapsed:.0f}s, "
            f"{destination.stat().st_size / 1e6:.0f} MB)"
        )

    meta = {
        "model": MODEL_ID,
        "layers": list(layers),
        "pooling": "masked mean over real residues (padding and BOS/EOS excluded)",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "frozen": True,
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"metadata -> {args.outdir / 'meta.json'}")


if __name__ == "__main__":
    main()
