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

Of 79 logical datasets, **4** have a held-out section carrying a usable
curated annotation. All four are Xenium and all four are skin.

**Corrected 2026-09-03.** This previously read "5 ... and 4 were measured
(`xhs1011` was lost to a shell bug)". The shell bug (§5) was real and is
fixed, but recovering the columns showed `xhs1011` was never the fifth
dataset: it has **no cell-label column at all**, and its only niche column is
`Xenium region number2` — the instrument's slide-region index, whose 10
categories are `'1'..'8'`, `'MISSING'` and `'psoriasis'`, with 747,233 of
1,553,516 non-null cells (48%) in `MISSING` and the one meaningful value a
DISEASE label. Scoring it would measure recovery of a technical slide-region
effect, the opposite of what the niche branch is for. So the bug was masking
an unusable dataset rather than costing a usable one, and it must now be
excluded explicitly — the fix makes these two columns reach the predicted
object for the first time, which would otherwise *add* two spurious rows.
The binding constraint is tighter than recorded, not looser.

```
sub-datasets in the corpus                        156
... with >=1 section in TERRA test                 27
... whose test sections carry a curated label       7   (5 logical, 4 usable)
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
  `L2_dist_broad_anatomy_Genital Tubercle`, …) — 121 names became 128 words,
  none of the 7 fragments matched a real column, and nothing in the log said
  so. Pass such lists as a quoted bash ARRAY (`mapfile -t` + `"${arr[@]}"`),
  never as a single space-joined string. Fixed in `_tier1eval_job.sh`.
  Note the fix has a sharp edge: those columns now REACH the predicted object,
  so anything technical among them must be excluded from scoring explicitly
  (see §3) — a silent omission became a silent inclusion.
- **`uns['squint']` is flattened with `str()` on write**, so nested values come
  back as `"{'cell': [30, 90], ...}"`. A bare `except Exception` around the
  parse turned that into a plausible-sounding "no codebook sizes available" and
  produced NaN utilisation across two full corpus runs.

---

## 6. Tier 1 outcome: the LR schedule is the win, and it moved the checkpoint

Run `corpus-holdout-tier1/20260902_221324_seed0`, 200,000 steps in 4.89 h
(baseline 6.9 h), against `corpus-holdout/20260827_082804_seed0`. Identical
data, split, budget and scoring code.

**The four changes did what the mechanism said they would.**

| | baseline | tier1 |
|---|---|---|
| RVQ level 1 used, cell / niche | 43/90, 54/90 | **90/90, 90/90** (util 1.000, every split) |
| adjacency share of the objective | 55.3% | **16.9%** |
| NB cell / neighbour recon loss | 704.7 / 773.2 | 680.4 / 721.7 (−3.5% / −6.7%) |
| adjacency RAW (weight removed) | 1.8722 | 1.9468 (+4.0%) |

Every term improved except the one deliberately down-weighted.

**The headline: the cosine schedule inverted which checkpoint is best.**
Under the baseline's constant 7e-4, `best` (step 40,000) beat `last`
everywhere. Under the schedule the LR at step 40,000 is still 6.414e-04 —
near peak — so `best` is no longer the annealed model and `last` is. Held-out
test, per dataset, level-0 codes:

| arm | branch | ΔNMI | ΔARI |
|---|---|---|---|
| last | cell | **+0.050 (87/87)** | **+0.046 (87/87)** |
| last | niche | +0.063 (15/17) | +0.045 (12/17) |
| best | cell | −0.025 (4/87) | −0.026 (18/87) |
| best | niche | +0.024 (13/17) | −0.030 (5/17) |

Under TERRA's own protocol (latent @ matched K, test): `last` cell NMI
0.391 → **0.482** (81/81) and niche 0.399 → **0.434** (15/15). `tier1/last`
is the strongest configuration measured so far on that protocol.

**Reconstruction improved on the test rung and REGRESSED on the val rung.**
R_test (xhs1022, Xenium skin, 120,889 cells) improved in 7 of 8 measures,
`last`/niche cell-wise most: 0.395 → **0.594**. But R_val (chr78, CosMx, a
DIFFERENT ASSAY carrying 2 distinct gene sets across its 11 sections) fell,
worst at `last`/cell 0.244 → **0.083**. Whether that is assay specialisation
or a chr78-specific panel effect is untested and is the first thing to check
before treating Tier 1 as a clean win.

**What did NOT move: batch integration**, exactly as predicted. iLISI on test
fell slightly (cell_emb 0.0496 → 0.0435), still ~14% of the 0.30 ceiling. The
encoder still receives no batch information — that is Tier 2. ASW did improve
(cell_latent under `last`, −0.250 → −0.118).

**What Tier 1 did not achieve:** it did not beat `baseline/best` on raw
code-based CELL identification (NMI 0.385–0.498, ARI 0.227–0.323 per dataset,
still the highest of the four configurations). Codebook expansion remains
ruled out by §1, so that ceiling is still the latent's class separability.

### A measurement trap the fix opened, in three places

Fixing the `--keep-obs-cols` quoting made the technical columns reach the
predicted object for the first time, and each label-consuming path discovers
labels independently. `Xenium region number` / `number2` therefore had to be
excluded in `compute_inference_metrics` (via the pool job), in `_tier0.py`
and in `_tier0c.py` — 112 spurious rows across FOUR datasets (xhs1000,
xhs1009, xhs1010, xhs1011) on the first Tier 1 pass, not just xhs1011. A
silent omission became a silent inclusion; grep for every label-discovery
site when changing what reaches `.obs`.

---

## 7. The chr78 reconstruction regression is an OOD-width effect, not assay specialisation

§6 flagged Tier 1's R_val regression (chr78, CosMx, `last`/cell cell-wise ρ
0.244 → 0.083) as untested. Tested now, from the existing full-cache predicts.
Three candidate causes ruled out **by measurement**:

- **Not the assay.** chr78 IDENTIFICATION is unchanged (ΔNMI −0.0004), and
  chl59 (+0.044) and shp75 (+0.046) — also non-Xenium validation datasets —
  both improved. The encoder generalises fine; the regression is decoder-only.
- **Not a masking bug.** Predicted mass inside each section's OWN panel is
  exactly **1.0000** in both runs, checked per section against that section's
  own `var`. An earlier reading of "3.5pp leakage" was an artefact of slicing
  20,000 rows that spanned THREE sections while applying one section's panel;
  it does not reproduce per section. Check `adata_batch_id` before assuming a
  row slice is one section.
- **Not chr78's internal panel split.** Its two gene sets are 169 genes each
  with a 165-gene intersection, and the regression is uniform across all 11
  sections (−0.12 to −0.18), not bimodal.

**What it is: the decoder's per-cell profile flattens on an unseen panel
width.** Measured per section, predicted profile CV against a true CV of
~5.0–5.8: CM008 1.495 → 1.253, CM013 1.965 → 1.374, CM017 1.663 → 1.317
(−16% to −30%). On the wide panel it moves the OTHER way — xhs1022 (4,947
genes) 1.504 → **2.100**, and ρ rises 0.269 → 0.281. Flatter within-cell
profile ⇒ lower cell-wise ρ; gene-wise ρ is unaffected by within-cell
flatness and indeed IMPROVED on chr78 (+0.015), which is the signature.

**Why that width:** panel width by split —

| split | min | p5 | median |
|---|---|---|---|
| train | **237** | 289 | 477 |
| validation | **169** | 169 | 946 |
| test | **252** | 252 | 4,948 |

**No training section has ≤200 genes.** chr78's 169 occurs only in validation.
Cosine annealing sharpens the decoder onto the training distribution, so
in-distribution reconstruction improves and extrapolation to a width nothing
in training covers degrades. A textbook annealing/OOD tradeoff.

**Settled: it is OOD widths only, not narrow panels.** `_narrowtest_job.sh`
ran `xhs1009` — 252 genes, the narrowest width in TERRA test, but IN
distribution because 4 training sections share it exactly — with a full cache
on both arms of both runs. Tier 1 improved cell-wise ρ in **13/13 sections on
both arms**: `best` +0.0914, `last` +0.0565.

The dispersion tells the same story from the other side. On the
in-distribution narrow panel `last` SHARPENS (CV 1.9 → 2.2 against a true
3.3), exactly as on the wide panel; on the OOD narrow panel it FLATTENS
(1.5 → 1.25 against a true 5.0). Same model, opposite direction, split purely
by whether that width appeared in training.

| rung | width | in training? | Δρ (`last`) |
|---|---|---|---|
| xhs1022 | 4,947 | yes | +0.031 |
| xhs1009 | 252 | yes (4 sections) | **+0.057** (13/13) |
| chr78 | **169** | **no** (floor is 237) | **−0.161** |

**Verdict: Tier 1 is adoptable as-is.** The regression is confined to the one
dataset below the training panel-width floor, and that dataset is entirely in
validation — it does not appear in TERRA test at all, so no reported test
number is affected. Every in-distribution dataset measured improves.

**The caveat that must be reported**, though: after annealing the model
extrapolates worse to panels narrower than anything it trained on. Applying
SQUINT to a new 100–200 gene panel should expect degraded reconstruction (not
degraded identification — that was unaffected). The corpus contains no panels
below 237 genes to learn from, so a fix would have to be training-side —
width augmentation, i.e. randomly subsetting wide panels during training so
the model sees narrow ones. Untested, and not required for the current
claims.

---

## 8. What "batch" means here, and what Tier 2 should actually target

`batch_key` is `dataset_batch` = **one tissue section**, and iLISI is computed
over it. The 187 test sections span **four organs** (skin 111, kidney 42,
pancreas 24, brain 10), so the global number partly asks a skin cell to have
kidney neighbours. Scoped per tissue (`_ilisiscope.py`, `cell_emb`, k=90,
50,000 cells, same settings as the metrics script):

| scope | sections | iLISI | ceiling | % of achievable |
|---|---|---|---|---|
| global | 187 | 0.0363 | 0.479 | 7.6% |
| skin | 111 | **0.2210** | 0.809 | 27.3% |
| brain | 10 | **0.2452** | 1.000 | 24.5% |
| pancreas | 24 | 0.1828 | 1.000 | 18.3% |
| kidney | 42 | 0.0950 | 1.000 | 9.5% |

**Both halves matter.** Within-tissue iLISI is **3–7× the global raw value**,
so the headline number really does overstate the problem — tissue separation,
which we want, is a large part of the shortfall. But at 9–27% of achievable,
sections of the *same organ* still do not mix well, so there IS a real
technical batch effect. **Tier 2 is justified, but the target is within-tissue
section mixing and the metric to report is per-tissue iLISI, not global.**

**The assay row is degenerate — do not quote it.** It reads 0.0000 on every
embedding, which looks like total assay separation. TERRA test is 186 xenium
sections against **1 cosmx section** (0.05M cells, 0.35% of test); in a
50,000-cell subsample that is ~175 cells, so a cell's 90 neighbours essentially
never contain one. Class imbalance, not a batch effect. A per-assay claim needs
a balanced subsample.

**A ceiling formula correction.** §5 records the iLISI ceiling as
`(k-1)/(n_batches-1)`. That holds only when `n_batches > k`; below it the
ceiling is 1.0. The correct general form is
`(min(k, n_batches) - 1) / (n_batches - 1)`. Applied unguarded to a
within-tissue scope of 10 sections the old formula returns 9.89, which is how
the bug announced itself. The 0.30 figure for 297 batches at k=90 is
unaffected.

### Tier 1 reduced technical nuisance where iLISI could not see it

The section-similarity regression, rerun on Tier 1's annealed checkpoint:

| coefficient | baseline | tier1 | |
|---|---|---|---|
| same tissue (niche) | 0.1327 | **0.1477** | +11% — biology, kept |
| same panel width (niche) | 0.0657 | **0.0497** | −24% — nuisance, reduced |
| same assay (niche) | 0.0224 | **0.0115** | −49% — nuisance, reduced |
| same panel width (cell) | 0.0770 | 0.0694 | −10% |
| same assay (cell) | 0.0116 | 0.0060 | −48% |

Biology-to-nuisance ratio (tissue ÷ panel width) on the niche branch:
**2.02 → 2.97, a 47% improvement**; on the cell branch 2.89 → 3.07. So Tier 1
*did* move integration in the right direction — pushing technical structure out
while holding or increasing tissue structure — it simply is not what a
section-level iLISI measures. Worth reporting alongside iLISI rather than
instead of it.

---

## 9. Weight decay is erasing the batch embedding, in every corpus run

Tier 2a trained cleanly (200,000 steps, 5.08 h, all 25 preflight checks) and
encoder FiLM armed correctly at `condition_dim=416`. But its FiLM generator
has **355 of 416 columns exactly zero**, and the decoder's batch embedding is
no better -- non-zero rows are **52/416 (baseline), 69/416 (Tier 1), 67/416
(Tier 2a)**. Roughly 85% of sections have no usable batch representation in
any of the three runs.

**This is not "those sections were never sampled".** A never-sampled row would
keep its `N(0, 0.02^2)` init, norm ~0.08 -- not zero. The rows were actively
driven there. Simulated directly: `Adam(lr=7e-4, weight_decay=0.001)` applied
to a parameter receiving zero gradient reaches **0.000000 within 1,000 steps**,
because the adaptive normalisation makes the decay step ~`lr` regardless of
the parameter's magnitude.

`BaseModel._build_optimizer` passes `self.parameters()` as ONE group, so decay
hits the batch embedding and the FiLM generator like everything else. With
~114 blocks over a 200,000-step run at ~3-4 sections per block, a given
section appears in roughly one block -- about 1,758 consecutive steps -- and
then receives ~198,000 steps of pure decay. Only the largest sections, and
those sampled near the end, survive.

**This is a strong candidate for the actual cause of the integration failure.**
The decoder covariate is the paper's batch-correction lever (Eq. 6) and it has
been structurally disabled for 85% of sections in every run measured. It also
explains why Tier 2a moved nothing: FiLM cannot condition on a batch whose
column is zero.

**The fix is a no-decay parameter group** for `batch_embedding` and
`conditioning_module` -- embeddings and normalisation-style parameters are
conventionally excluded from weight decay for exactly this reason.
`configure_optimizers` already builds the optimizer in one place, and a
commented-out two-group version is sitting in that method's history.

**Do not evaluate Tier 2a as a test of encoder conditioning.** It measures a
model whose conditioning was erased. Fix the parameter grouping and rerun
before spending the ~6 h evaluation.

**A measurement note.** "Exactly zero" and "never updated" are different
claims, and only the second was interesting. Checking against the INIT value
rather than against zero is what separates them -- a parameter at exactly 0.0
under Adam+weight-decay is evidence of decay winning, not of absence.

---

## 10. The batch embedding is repaired, and integration still did not move

`corpus-holdout-tier1-nodecay`, one change from Tier 1. The repair is total:

| run | non-zero rows | median row norm |
|---|---|---|
| baseline | 52/416 | 0.0000 |
| tier1 | 69/416 | 0.0000 |
| tier2a | 67/416 | 0.0000 |
| **tier1-nodecay** | **416/416** | **0.3293** |

0.3293 is well above the `N(0, 0.02^2)` init norm of 0.08, so all 416 rows
genuinely learned rather than merely surviving. The decoder covariate — the
paper's batch-correction lever (Eq. 6) — is functional for the first time in
any corpus run.

**And integration is flat.** Per-tissue iLISI on `cell_emb`, held-out test:

| scope | tier1 | nodecay | Δ |
|---|---|---|---|
| skin (111 sections) | 0.2210 | 0.2364 | +0.015 |
| kidney (42) | 0.0950 | 0.1092 | +0.014 |
| brain (10) | 0.2452 | 0.2470 | +0.002 |
| pancreas (24) | 0.1828 | **0.1544** | **−0.028** |
| global (187) | 0.0363 | 0.0378 | +0.002 |

Inconsistent in sign and within noise of each other; `neighborhood_emb` mostly
moved the *wrong* way (skin −0.009, kidney −0.011). Repairing a structurally
broken mechanism bought nothing on the axis it exists to serve.

**Why this is architecturally unsurprising, and what it implies.** The decoder
covariate lets the DECODER absorb batch effects, which only *indirectly*
relieves the encoder. Nothing in the objective rewards a batch-free latent, so
the encoder has no reason to produce one. The same argument applies to Tier
2a's encoder FiLM: it gives the encoder the MEANS to subtract batch out, but
still no MOTIVE. **On this evidence the expected outcome of Tier 2a-nodecay is
also flat**, and what would actually force batch-invariance is a term in the
OBJECTIVE — the adversarial head (Tier 2b) or an MMD batch term, both of which
already exist in `loss/` and are switched off by `_patch_dual_no_batch_int` in
the reference stack.

**What the repair did buy**, held-out test unless noted:

- **Cell identification on `best`: NMI 0.3841 → 0.4144, +0.030, winning
  87/87 rows.** The largest identification gain of any change so far.
- **R_val reconstruction, the OOD narrow panel (chr78)**: `last`/cell
  0.0832 → **0.1222**, `last`/niche 0.0749 → **0.1157** — roughly +50%
  relative, partially recovering the §7 regression. Consistent with the
  mechanism: a held-out section takes the mean of trained embeddings, and that
  mean is now an average over 416 real rows rather than 69.
- Training objective ~3% worse across every term (total 1724 → 1775), which
  is the cost of the decoder actually using a per-batch signal.
- `last` degraded on identification (cell NMI −0.018, 0/87), so the arms
  flipped again: `nodecay/best` is now the strongest cell-identification
  configuration measured.

**Do not read the flat iLISI as "the fix was pointless."** The embedding was
broken and is now correct; every number above is measured on a model that
finally has the mechanism the paper describes. The finding is narrower and
more useful: *batch conditioning at the decoder does not de-batch the latent*,
so integration needs an objective term, not more conditioning capacity.

---

## 11. Three epochs is the best model, and integration is now definitively an objective problem

`corpus-holdout-nodecay-3ep` — 562,687 steps (3.00 epochs), 13.4 h, peak 82.9
GB. One change from `corpus-holdout-tier1-nodecay`: the budget, with the LR
horizon re-synced so the cosine schedule spans the real run.

**Note on arms:** in this run the FINAL checkpoint (step 562,680) also has the
lowest `val_loss` (1720.891), so `best` and `last` resolve to the same
weights. Duplicated rows in its tables are correct, not a collision — but no
arm comparison is available for it.

### It is the strongest model measured on every identification metric

Per dataset, held-out test, level-0 codes:

| | tier1/best | nodecay/best | **ep3** |
|---|---|---|---|
| cell NMI | 0.3841 | 0.4144 | **0.4178** |
| cell ARI | 0.2111 | 0.2262 | **0.2269** |
| niche NMI | 0.3750 | 0.3606 | **0.3941** |
| niche ARI | 0.1996 | 0.1862 | **0.2451** |

The niche gains are the story: **+0.034 NMI (13/17 rows) and +0.059 ARI** over
nodecay. That is where the epoch-end trace predicted it — NB neighbour fell
806.3 → 774.9 → 713.9 across the three epochs, −11.5%, the largest movement of
any term in any run, while `val_loss` bounced and hid it (§4's warning, in
action: the total is a weighted sum whose components move in opposite
directions and is not a learning curve).

Under **TERRA's protocol** (latent @ matched K) ep3 leads both branches: cell
**0.4874** (tier1/last 0.4818, nodecay/last 0.4750), niche **0.4544** (0.4337,
0.4304).

**And cell NMI enters the paper's range for the first time.** Per dataset:

| dataset | cell NMI | cell ARI | niche NMI | niche ARI |
|---|---|---|---|---|
| xhs1022-2 | **0.5211** | 0.3182 | 0.4827 | 0.2849 |
| xhs1022-1 | 0.5016 | 0.2838 | 0.4472 | 0.2700 |
| xhs1022-3 | 0.5003 | 0.2987 | — | — |
| xhs1000 | 0.4733 | 0.2283 | 0.4597 | 0.2980 |
| xhs1009 | 0.4151 | 0.2165 | 0.3322 | 0.2143 |
| xhs1010 | 0.3951 | 0.2169 | 0.3655 | 0.2273 |

Paper: cell NMI 0.494–0.604, ARI 0.204–0.339; niche NMI 0.397–0.702, ARI
0.172–0.410. Three of six datasets now clear the cell-NMI floor, cell ARI sits
inside the range throughout, and niche ARI (0.214–0.298) is inside as well.

### Integration is now definitively an objective problem, not a capacity or conditioning one

Per-tissue iLISI, `cell_emb`, held-out test:

| tissue | tier1 | nodecay | ep3 |
|---|---|---|---|
| brain | 0.2452 | 0.2470 | 0.2105 |
| kidney | 0.0950 | 0.1092 | 0.0893 |
| skin | 0.2210 | 0.2364 | 0.2109 |
| pancreas | 0.1828 | 0.1544 | 0.1717 |

Three of four DOWN, and the global number too (0.0378 → 0.0345). Four
interventions have now failed on this axis:

1. rebalancing the objective and fixing the codebook (Tier 1) — flat
2. encoder FiLM conditioning (Tier 2a) — flat
3. repairing the erased batch embedding, 69/416 → 416/416 (nodecay) — flat
4. tripling the training budget (ep3) — slightly worse

**Nothing in the objective rewards a batch-free latent**, and conditioning
capacity, a working covariate and more optimisation are all substitutes for a
term that is not there. `_patch_dual_no_batch_int` in the reference stack
removes `adversarial_batch_loss`, `mmd_batch_loss` and `mmd_prior_loss`
outright; both remaining candidates already exist in `loss/`. Adding one is
the only untried lever, and after four failures it is also the only one worth
trying.

### One regression, and it is the §7 effect intensified

R_val reconstruction on chr78 (CosMx, 169 genes, below the 237-gene training
floor) collapsed: `best`/cell 0.2451 → **0.0879**, niche 0.2576 → **0.0411**.
More training sharpens the decoder further onto the training panel-width
distribution, so extrapolation below that floor degrades further — the same
mechanism as §7, now with 3x the annealing to sharpen it. Test-rung
reconstruction is unaffected (cell 0.3490, niche 0.7078) and chr78 is
validation-only, so no reported test number changes. But the caveat hardens:
**applying this model to a panel narrower than ~237 genes should expect poor
reconstruction**, and the effect grows with training rather than washing out.

**Capacity remains undiagnosable and unindicated.** The oscillation-on-fixed-
data signature (val_loss range 137 over 562,687 steps) is a data-order
signature, not a capacity ceiling, and identification improved by adding
epochs rather than parameters. Codebook expansion stays ruled out (§1), now
also by ep3: level 1 is 90/90 at perplexity 0.95–0.98 and identification moved
for unrelated reasons.

---

## 12. Compute nodes can execute stale source, and it looks like a logic bug

Three Tier 2b calibration sweeps failed with `mmd_batch_loss` returning `None`
to the loss dispatcher. That is impossible for a function whose every exit
returns a Tensor, and it did not reproduce on the head node with the exact
failing shapes (512x256 target, 8 sections, the real 416-entry tissue map).

The cause was not in the code. Introspecting the callable from inside the
failing job:

| | `return` count | diagnostic guard present |
|---|---|---|
| head node | **7** | **yes** |
| compute node, same path | **4** | **no** |

Same absolute path, different content, **thirty minutes after the write** — so
not a write race. The compute node was serving a cached copy new enough to
accept the `mmd_group_labels` keyword (hence no unexpected-kwarg error) and
old enough to lack the guard added later.

**What made this expensive.** Every symptom pointed at logic, so three
successive explanations were constructed and all were wrong — numerical
divergence at high weight, failure on the first batch, then weight-dependence.
Two of those were reinforced by a second problem: **`#BSUB -o` APPENDS**, so
three sweeps' output accumulated in one file and stale tracebacks read exactly
like fresh ones. The arm-status line was showing two `exited with` entries for
one arm, which is the tell that was there and missed. Most `_*_job.sh` in this
tree use `-o`; the Tier 2b jobs now use `-oo`/`-eo`.

**The guards now in place.** `_codecheck.py` prints the git HEAD, the package
path, and a SHA over the whole source tree as that node reads it, and
`--expect module:token` asserts a token is present in a module's source from
the compute node's view. Both Tier 2b jobs refuse to start if it fails, so a
run that would execute different code than the working tree dies instead of
producing a confusing result. Compare the printed `src sha` across array arms:
they must match.

**The general lesson, worth applying beyond this branch.** When a result is
impossible given the code, verify that the code that ran IS the code on disk
before constructing a mechanism that explains the impossible. The check costs
seconds; the alternative cost three cycles here.

