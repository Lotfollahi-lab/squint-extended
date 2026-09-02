# Corpus evaluation: what was measured and what it means

Findings from evaluating the first corpus-scale run
(`artifacts/hst_corpus_110m/corpus-holdout/20260827_082804_seed0`, 200,000
steps = 1.07 epochs over 96M training cells). Written down because several of
these overturned an earlier reading, and re-deriving them costs GPU hours.

Evidence lives in the untracked probes, per the project convention:
`_tier0_partA.json` (identification per dataset), `_tier0_partB.json` (masked
reconstruction), `_tier0c.json` (matched-K clustering), `_evalscope.json`
(scope + label curation), `_splitspec.json` (the TERRA split).

---

## 1. Codebook size is NOT the identification bottleneck

**The concern.** TERRA's published protocol clusters embeddings and matches the
cluster count to the annotation: *"The base resolution was chosen so that the
number of clusters matched the number of ground-truth classes with >100
cells."* SQUINT scores 30 fixed codes against annotations carrying 8–98
classes, so the two sit side by side under different rules. On synthetic data
where the embedding separates the truth perfectly and the codes are a *perfect*
30-way coarsening of 96 true classes, ARI is capped at 0.462 against 0.988 at
matched K — so the concern looked serious.

**The measurement** (`_tier0c.py`): three representations of the same cells,
increasing in freedom — codes at fixed K, quantised embedding clustered at
matched K, continuous latent clustered at matched K.

**The result: giving the model matched K does not buy identification.**

| branch | | codes (fixed 30) | latent @ matched K |
|---|---|---|---|
| cell (median K=56) | NMI | 0.451 | 0.471 |
| | ARI | **0.239** | 0.168 |
| niche (median K=21) | NMI | 0.390 | 0.469 |
| | ARI | 0.246 | 0.241 |

Matched-K wins NMI in 72/96 groups but ARI in only **6/96**. The synthetic
prediction — that the gap should grow with the annotation's class count — is
refuted: ΔARI goes **−0.059, −0.044, −0.106** across K bands 20–40, 40–70,
70–200. It gets worse with more classes, not better. Consistent across all four
datasets independently.

**Why the synthetic was wrong.** It assumed the latent cleanly separates the
true classes, in which case extra clusters resolve real structure. It doesn't.
Extra clusters fragment a latent that never separated those classes — NMI
tolerates that, ARI punishes it.

**Consequence: do not prioritise codebook-expansion experiments for
identification.** There is no headroom to buy. The binding constraint is how
well the latent separates annotation classes at all, which points at the
training config (§4), not at `codebook_size`.

### Two findings that came free

- **"No clustering" holds up.** SQUINT's selling point is scoring codes
  directly. The 30 fixed codes match or beat KMeans at matched K on the same
  embeddings — better ARI in 90/96 groups. Evidence *for* the architecture.
- **Quantisation costs NMI, the codes recover it.** latent → quantised loses
  0.056 NMI (cell) / 0.121 (niche), yet `codes_L0` beats `quant@K` in 79/81 and
  15/15 groups. Clustering ~1,350 code centroids at K=56 partitions them
  arbitrarily; the code IDs carry the partition the model actually learned.

**Caveat.** Four logical datasets, all Xenium skin; the 96 rows are ~24
annotations each, so not independent. KMeans, not Leiden (`leidenalg` is not
installed) — this is ours-vs-ours on identical cells, NOT a reproduction of
TERRA's absolute numbers.

---

## 2. "Dataset" has three meanings, and only one is right

| unit | count |
|---|---|
| tissue sections (one `.h5ad`) | 636 |
| silver subdirectories | 156 |
| **logical datasets** (shared numeric id) | **79** |

The convention is `<prefix><id>[-shard]-<N>b_<M>p`, and **`Nb` is the batch
count for the whole id, not the directory**. `xhs1022-1-55b_1p`, `-2-`, `-3-`
all declare 55 batches because they are three shards of ONE dataset. 12 ids
span several directories; one spans 21. Counting subdirectories overstates
dataset diversity — an earlier report said "6 datasets" when three of the six
were shards of `xhs1022`.

**The corpus is 636/636 *homo sapiens*.** There is no mouse data. The `m`
prefix is MERFISH (`c`=CosMx/CARTANA, `x`=Xenium, `s`=STARmap); `mhb83-1b_1p`
is MERFISH-Human-Brain. Composition: 20 tissues, 5 assays, all human.

---

## 3. Held-out annotation coverage is the binding evaluation constraint

Of 79 logical datasets, **5** have a held-out section carrying a curated
annotation, and **4** were measured (`xhs1011` was lost to a shell bug, §5).
All four are Xenium and all four are skin.

```
sub-datasets in the corpus                        156
... with >=1 section in TERRA test                 27
... whose test sections carry a curated label       7   (5 logical)
```

No config change fixes this. Broadening it means annotations on held-out
sections from cosmx/merfish/starmap and from tissues other than skin — a
curation task with a long lead time, worth starting in parallel with training
work.

**The scoring protocol matters as much as the coverage.** A cell-count-weighted
average over all 18 niche vocabularies gives NMI 0.349; per dataset against one
curated annotation (chosen by *coverage*, never by score) it is 0.387–0.432,
with three of five clearing the paper's floor of 0.397. The pooled average
double-counts cells 2.6× (a cell with five niche columns contributes five
times) and weights datasets by how many synonymous annotation columns they
happen to ship — `xhs1010` carries **62** cell-annotation columns spanning NMI
0.057–0.481.

---

## 4. Where the real deficits are, with config evidence

Measured against the paper, per dataset, held-out test:

- **Identification** — cell-type ARI 0.204–0.323 vs the paper's 0.204–0.339
  (parity); NMI 0.448–0.498 vs 0.494–0.604 (overlaps only at the bottom).
  Niche NMI 0.387–0.432 vs 0.397–0.702.
- **Reconstruction** — roughly half the paper on the cell branch (masked
  cell-wise 0.336 vs 0.634; gene-wise 0.183 vs 0.362), 79% on the
  neighbourhood branch. **Masking made this worse, not better** (§5), so the
  gap is in the model, not the metric.
- **Integration** — iLISI 0.049 against a structural ceiling of 0.30 for 297
  batches at k=90 neighbours, i.e. ~17% of what is achievable.

Four things in `user_specified_config.yaml` that plausibly explain the last two:

| finding | value |
|---|---|
| adjacency BCE dominates the objective | 1,872 of 3,385 = **55%**, and it was the one term getting *worse* across training |
| level 1 of each RVQ can never revive a dead code | `ResidualVQ_Squint` passes `threshold_ema_dead_code if i == 0 else 0`, so revival is armed on level 0 only. Level 1 uses 45/90 and 55/90 codes at perplexity_norm 0.25-0.43 |
| the commit signal is drowned, not absent | `commitment_weight: 0.0` is deliberate (the lucidrains-internal loss); the real signal is the external `mse_commit_loss_{cell,niche}` at `wt_commit_* = 1.0`, which measures 0.1% of the loss *because* adjacency is 55% of it |
| the encoder gets no batch information | `adversarial_alpha: 0.0`, `adversarial_batch_dim_request: False`; decoder covariate only, **16 dims for 635 batches** |
| no LR schedule | plain Adam, constant 7e-4, 200k steps, no warmup or decay |

Also `codebook_diversity_loss_weight` is 10.0 on cell and **0.0** on niche —
almost certainly unintended, and niche's level 1 is *better* used without it.

**Two of these were initially mis-attributed, and the corrected reading changes
the fix.** The half-dead level-1 codebooks are not caused by
`commitment_weight: 0.0` — that value is a documented convention, with the
external `mse_commit_loss_{cell,niche}` carrying the commit signal at weight
1.0. The cause is `hierarchical_vq.py`: dead-code revival is hard-wired to
level 0, reasoned as avoiding spurious revival from sparse usage *at init*,
but the switch is permanent and over 200,000 steps it simply costs half the
codebook. So the fix is per-level revival (`dead_code_all_levels`), not a
commitment weight; and the commit terms are raised 6.7x for free by the
adjacency reduction, without touching `wt_commit_*` — one lever, so the effect
stays attributable.

Note what the codebook fix is *not* expected to buy: per §1, effective
codebook size does not bind identification, so NMI/ARI should not move. Finer
quantisation should help **reconstruction**, which is the actual deficit.

### Tier 1: these four, as one run

`_patch_tier1` in `run_squint.py`, registered as the variant
`corpus-holdout-tier1`, built by calling `corpus-holdout`'s own builder so the
diff against it is exactly the patch. Same 200,000-step budget, same data,
same split — the point is an attributable delta, which a simultaneous budget
change would destroy.

| change | from | to |
|---|---|---|
| `wt_adj_reconstr` | 1000 | 150 (~16% of the objective) |
| RVQ dead-code revival | level 0 only | all levels (`dead_code_all_levels=True`) |
| LR | constant 7e-4 | 2,000-step warmup, cosine decay to 0.05x |
| `codebook_diversity_loss_weight` | 10.0 cell / 0.0 niche | 0.0 both |

Diversity is symmetrised *downward* because the only evidence available points
that way (the branch without the term used more of its second codebook), and
because it removes a term from an objective whose headline problem is one term
dominating. It is the weakest-motivated of the four; `diversity_weight=10.0`
symmetrises upward instead.

The LR schedule needed code: `BaseModel.configure_optimizers` returned a bare
optimizer, so this model had never handed Lightning a scheduler. It now returns
the `{"optimizer", "lr_scheduler"}` mapping when `lr_schedule='cosine'`, and a
bare optimizer otherwise — so every variant predating the keyword is unaffected.
Both new paths are covered by `tests/test_tier1_config.py`.

`best` (step 40,000) vs `last` (step 200,000): `best` wins identification and
reconstruction on every row; `last` wins batch integration on the quantised
embeddings across all splits. `best` had seen 22% of training cells, 107/417
sections, 23/40 panels, 15/18 tissues, and all three assays present in training.

---

## 5. Measurement traps found the hard way

- **Reconstruction cell-wise Pearson is inflated by structural zeros.** On a
  169-gene panel widened to a 9,574-gene vocabulary, ~98% of each cell's vector
  is exact zero on both sides; after centring those correlate perfectly.
  Masking to measured genes lowers cell-wise by 0.02–0.22 (worst on narrow
  panels) and gene-wise by only 0.013–0.036 — gene-wise was already effectively
  masked, because a constant-zero gene yields `nan` and is dropped by
  `vec[np.isfinite(vec)]`.
- **iLISI has a structural ceiling of (k−1)/(n_batches−1)** = 0.30 at 90
  neighbours and 297 batches, not 1.0. Never compare it across settings with
  different batch counts. The paper's values exceed 1 and are not on the
  `scib_metrics` [0,1] scale at all.
- **MMD is quadratic in batch count** and `--mmd-max-pairs` defaults to no cap:
  297 batches is 43,956 pairs, ~8 h. It is a *mean over pairs*, so 200 sampled
  pairs give the same answer in minutes.
- **`--keep-obs-cols` was word-split by the shell**, silently dropping every obs
  column whose name contains a space (`Xenium region number2`,
  `L2_dist_broad_anatomy_Genital Tubercle`, …). Cost one whole dataset
  (`xhs1011`) from the evaluation. Pass such lists via a file, not shell words.
- **`uns['squint']` is flattened with `str()` on write**, so nested values come
  back as `"{'cell': [30, 90], ...}"`. A bare `except Exception` around the
  parse turned that into a plausible-sounding "no codebook sizes available" and
  produced NaN utilisation across two full corpus runs.
