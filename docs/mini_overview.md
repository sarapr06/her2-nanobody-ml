# The AI nanobody design pipeline, and where each model sits

## The pipeline

Going from "we want a binder against HER2" to "here are 20 sequences worth ordering"
is five stages. Each one is a filter: it takes a larger, cheaper set of candidates and
hands the next stage a smaller, better-characterised one. The expensive
structure-based tools sit late precisely because they cost too much per candidate to
point at a naive library.

```
(0) TARGET DEFINITION   HER2 seq/structure -> chosen epitope ("hotspot")
(1) GENERATION          (a) wet lab: immunize alpaca, pan, NGS  <-- this test
                        (b) de novo: generate backbones in silico
                        -> 10^2..10^6 candidate VHH sequences
(2) CHEAP RANKING       score every candidate, sequence only, no folding
                        -> ranked list; keep top ~1-5%   <-- Tasks 1 and 2
(3) STRUCTURAL TRIAGE   fold survivors, dock, score interface
                        -> complex models, affinity estimates
(4) DEVELOPABILITY      stability, aggregation, immunogenicity -> ~10-50 seqs
(5) ORDER AND TEST      express, purify, SPR/BLI -> real K_D
                        `-----> feeds back into (2)
```

Two things about this shape matter more than any individual model.

**The funnel is economic, not intellectual.** Stage 2 is cheap and inaccurate; stage 3
is expensive and better; stage 5 is the only ground truth. You place a model by asking
what it costs per candidate and how much it narrows the field.

**The loop closes at stage 2.** Measured affinities from stage 5 are the training data
for the next round's ranker. That is the stage this coding test is about: our model
consumes a noisy enrichment proxy (`target_IE_score`) rather than K_D, which is
exactly the data a first round actually produces.

---

## Where each model fits

Grouped by the job they do, not by lab or architecture.

### Structure prediction — "what shape is this, and does it dock?"

**AlphaFold-Multimer** (2021). *In:* two or more sequences. *Out:* a structure for the
complex, with per-interface confidence. The workhorse for asking whether a designed
VHH and its target form a plausible interface. Stage 3.

**AlphaFold3** (2024). *In:* sequences plus, unlike its predecessors, nucleic acids,
ligands and covalent modifications. *Out:* a joint structure for the whole assembly.
Same stage-3 job as Multimer, broader input vocabulary.

**ESMFold2**. *In:* a single sequence, no MSA. *Out:* a predicted structure. The trade
is speed for accuracy: MSA-free folding is fast enough to run over thousands of
candidates, which makes it the tool for a *first* structural pass at the top of stage 3
where AlphaFold would be unaffordable.

**Rosetta** (Nature Methods 2020 for the modern suite). *In:* a structure, plus a
protocol. *Out:* energies, minimised structures, docked poses, designed sequences.
Physics-based rather than learned, and the oldest thing on this list. Still the default
for stage-4 questions — relax this interface, estimate a ΔΔG for this mutation — and
for scoring the outputs of the generative models below.

### Representation — "turn a sequence into numbers"

**ESM / ESMC** (the family used in Task 1). *In:* a sequence. *Out:* a per-residue
embedding, and a likelihood for each position. Two distinct uses: as a frozen feature
extractor for a small supervised head (what Task 1 does), and as a zero-shot fitness
proxy, where a low-likelihood mutation is a guess at "this protein is less viable".
Stage 2, and the cheapest thing here — no structure, one forward pass.

### Generative design — "invent a binder that wasn't in any library"

**RFdiffusion** (2023; antibody-specific version 2025). *In:* a target structure and an
epitope to hit. *Out:* novel backbones designed to bind there, with sequences assigned
by a downstream design step. This is the stage-1(b) alternative to immunizing an
animal. The 2025 antibody work extends it to designing CDR loops on a fixed framework,
which is the nanobody-relevant case.

**BoltzGen** (2025). *In:* a target specification. *Out:* generated binder designs.
Occupies the same stage-1(b) slot as RFdiffusion.

**Germinal** (Nature Biotechnology 2026). *In:* a target and epitope. *Out:* de novo
antibody/nanobody designs. Presented as an end-to-end pipeline rather than a single
model, i.e. generation plus filtering in one workflow — stages 1(b) through 4.

### Affinity prediction — "how tightly, not just whether"

**Boltz-2** (2025). *In:* a complex specification. *Out:* a structure **and** a binding
affinity estimate. Notable because predicting affinity rather than just geometry is
what stage 3 actually needs: a ranked shortlist, not a pile of plausible poses.

**BoltzProt-1** (2026). *In:* protein sequence/structure input. *Out:* a general-purpose
protein model from the Boltz line; no separate public repository, released via the
`boltzgen` codebase and an API.

---

## Two honest caveats

**On the newest three.** The specific claims above for **Germinal**, **BoltzProt-1** and
**ESMFold2** are the least certain — they are recent enough that I would verify the
input/output description against the linked papers before defending any of it in
detail. The stage each occupies is the part I am confident about; the exact interface
is worth a five-minute check.

**On what none of them do.** Every model here reasons about structure, sequence
plausibility or affinity. None predicts what actually kills nanobody programmes:
expression yield, aggregation on storage, polyreactivity, or immunogenicity in a human
patient. Stage 4 is still mostly heuristics and assay data, and stage 5 is still the
only thing that settles an argument.
