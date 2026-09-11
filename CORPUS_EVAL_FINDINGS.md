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

**Which node, confirmed.** With the gate in place the sweep resolved it
completely: arms 1-2 reported `src sha cbdef8c29b8968b9` matching the head
node and ran; arms 3-5 reported STALE SOURCE and refused. All three of those
had been scheduled onto **`farm-gpu0307`**, and across every earlier sweep the
same array indices failed. So the apparent weight-dependence was array index
-> host assignment, nothing more. Touching the tree did not clear that node's
cache, so both Tier 2b jobs now carry `hname!='farm-gpu0307'` in their `-R`
select clause. The exclusion is a stopgap for one host; the gate is the real
safeguard and catches any other node.

**The general lesson, worth applying beyond this branch.** When a result is
impossible given the code, verify that the code that ran IS the code on disk
before constructing a mechanism that explains the impossible. The check costs
seconds; the alternative cost three cycles here.

---

## 13. A 2-hour iteration loop for integration, instead of 20

The full cycle is ~14 h training plus ~6 h evaluation per arm. For the one
question Tier 2b asks -- did within-tissue batch mixing improve? -- nearly all
of that is wasted:

| stage | cost | needed for iLISI? |
|---|---|---|
| predict, 12 shards over 297 sections | 17 min wall | partly |
| pool 90 GB | 46 min | **no** |
| metrics: identification over 1,128 combos | **~5 h** | **no** |
| iLISI itself | minutes, 50,000 cells | yes |

`_fastint.py` reads the UNPOOLED `eval_A_{arm}_sh*` shards and computes
per-tissue iLISI directly, at the same k=90 and scib scaling as
`compute_inference_metrics.py`. **7 minutes**, and it reproduces the full
pipeline's ep3 numbers to within 0.004:

| tissue | full pipeline | fast probe |
|---|---|---|
| brain | 0.2105 | 0.2147 |
| kidney | 0.0893 | 0.0909 |
| pancreas | 0.1717 | 0.1679 |
| skin | 0.2109 | 0.2114 |

`_fastloop_job.sh` chains train -> predict -> iLISI at 50,000 steps over a
balanced 46-section test rung (`_fastrung.json`), with **arm 1 as a wt=0
control** on the identical budget and rung -- without which a shift cannot be
told from the effect of shortening the run. About 2 h per arm, both in
parallel.

**Two design errors on the first attempt, both worth recording.** The predict
stage was killed at 389 GB and 675 GB because (a) it ran all sections in ONE
pass, reintroducing exactly what the main evaluation's 6-way sharding exists
to prevent -- a single-pass corpus predict had already died once at 424 GB --
and (b) it selected the LARGEST sections per tissue, when iLISI subsamples to
50,000 cells and therefore wants many sections and FEW cells. Selecting
smallest-first under a cell budget cut the rung from 10.57M cells to 2.6M, and
it is now sharded one tissue per shard so peak memory is bounded by the
largest tissue (brain, 1.9M cells) rather than the whole rung.

A third, and the one that would have silently invalidated the result: the run
directory is `<YYYYMMDD_HHMMSS>_seed<n>` at SECOND granularity with no name
override, so two array elements starting in the same second get the
**identical** directory. Both fastloop submissions did exactly that -- both
arms reported `Run directory: .../20260907_232921_seed0` -- which means two
models sharing one `checkpoints/` directory and one set of predict outputs.
The first submission's OOM masked it. Fixed by staggering the arms 90 s and by
asserting after training that exactly one training log names the chosen
directory; the arms also now read the directory from their OWN log rather than
`ls -1dt`, which is required for running the fast loop alongside the 14 h job
since both use the same variant.

A fourth, cosmetic but costly: the train stage piped through `| tail -30`, which
emits nothing until the pipeline closes, so a healthy 2-hour run looked frozen
in the LSF log while `tee` wrote the real one. Replaced with a line-buffered
grep on heartbeats and error signatures.

**What it is not.** The reduced scope changes the iLISI ceilings, so its
numbers compare the two arms against each other and NOT against sections
8/10/11. Identification, reconstruction, codebook and matched-K all still need
the full pipeline. Use it to decide whether a run deserves 14 h, not to report
final numbers.

---

## 14. MMD shows no integration signal at 50,000 steps

The fast loop ran both arms to completion on distinct run directories
(`20260907_234400` and `20260907_234532`, 92 s apart), identical 41-section
rung, identical budget, the only difference being `wt_mmd_batch`.

| tissue | sections | wt=0 | wt=400 | Δ |
|---|---|---|---|---|
| **cell_emb** — the branch MMD targets (`z_mlp`) | | | | |
| brain | 8 | 0.2043 | 0.1818 | **−0.023** |
| kidney | 11 | 0.1814 | 0.1935 | +0.012 |
| pancreas | 8 | 0.3930 | 0.3551 | **−0.038** |
| skin | 14 | 0.4556 | 0.4538 | −0.002 |
| | | | mean | **−0.013**, 1/4 improved |
| **neighborhood_emb** — not directly targeted | | | | |
| brain | 8 | 0.0826 | 0.1034 | +0.021 |
| kidney | 11 | 0.1181 | 0.1132 | −0.005 |
| pancreas | 8 | 0.2473 | 0.2844 | +0.037 |
| skin | 14 | 0.3314 | 0.3325 | +0.001 |
| | | | mean | +0.014, 3/4 improved |

**This reads as noise, not effect**, for three reasons. The two embeddings move
in OPPOSITE directions. The branch MMD actually acts on — `cell_emb`, since
`mmd_target = z_mlp[:batch_size]` — is the one that got WORSE, while the
untargeted branch improved. And **skin, the most reliable estimate** (14
sections, the largest cell pool, and 111 sections in the full scope), is flat
on both at −0.4% and +0.3%.

**What this does not establish.** 50,000 steps is 8.9% of the 3-epoch budget,
at near-peak LR, and there is no repeat arm to estimate run-to-run noise — the
±0.02 swings on brain and pancreas, each with only 8 sections, are plausibly
just that. MMD^2 *was* being minimised during calibration (0.0158 → 0.0087
from wt=100 to 400), so the term works as a term; it simply does not
translate into iLISI here.

**Where that leaves integration.** Five interventions have now failed to move
it: the Tier 1 objective rebalance, encoder FiLM, repairing the erased batch
embedding, tripling the budget, and now a scoped MMD objective term. The
remaining untried option is the adversarial head, which has known design
problems at this scale (a 416-way classifier sees ~8 classes per block, and
`wt_adv_batch=150` was calibrated for two batches). A defensible alternative
is to report integration as a documented limitation with five negative results
behind it, which is a stronger statement than an unexplained gap.

---

## 15. iLISI was the wrong metric, and on the right one integration DID move

**The objective, stated properly:** sections of the same tissue should get the
same codes, whatever the panel width, dataset or assay. That is a
code-DISTRIBUTION question, not a neighbourhood-mixing one, and iLISI answers
the wrong one twice over (§14 and below).

`_codeagree.py` measures it directly: per-section distribution over composite
RVQ codes, pairwise similarity = 1 - Jensen-Shannon, then

* **cross-panel agreement** = mean similarity for same tissue, DIFFERENT panel
  — the quantity to maximise, in absolute terms
* **TRANSFER** = that divided by same-tissue-same-panel — scale-free, but see
  the warning below
* **SEPARATION** = same-tissue / different-tissue — must stay HIGH, or a model
  that gave every tissue identical codes would score perfectly

All 636 sections, CPU only, ~4 min per run.

### Niche branch

| run | same panel | **diff panel** | TRANSFER | SEPARATION | xfer assay |
|---|---|---|---|---|---|
| baseline/best | 0.5119 | 0.3114 | 0.6084 | 2.9030 | 0.3424 |
| baseline/last | 0.5341 | 0.3429 | 0.6421 | 2.6486 | 0.4037 |
| tier1/last | 0.4704 | 0.3075 | 0.6538 | **3.7806** | 0.3586 |
| **nodecay/best** | 0.5235 | **0.3609** | 0.6895 | 3.7704 | 0.3110 |
| ep3/best | 0.4484 | 0.3318 | **0.7400** | 3.7526 | 0.3845 |

**Two real gains, invisible to iLISI.** Cross-panel agreement peaks at
**nodecay/best, 0.3609 — up 15.9% on baseline/best's 0.3114**; and tissue
SEPARATION jumped at Tier 1 (2.90 → 3.78) and held. So the objective did
improve, and the batch-embedding repair is what did it.

**But do not read TRANSFER alone.** ep3 has the highest ratio (0.7400) while
having the LOWEST same-panel similarity (0.4484) and a cross-panel value
BELOW nodecay's. The ratio rose because the denominator fell. This is the same
class of error as the iLISI ceiling: a normalised number that moves for
reasons unrelated to the thing being claimed. Report the absolute alongside it.

### Cell branch went the other way

Cross-panel agreement is highest at **baseline/best (0.5066)** and declines to
0.4430 at ep3. So the interventions traded cell-branch cross-panel agreement
for niche-branch agreement and tissue separation.

### Assay is the dominant nuisance, not panel

Transfer across assay is **0.31-0.55** against 0.61-0.74 for panel and
0.78-0.90 for dataset, on every run and both branches. Dataset identity barely
matters; assay matters most. This contradicts the additive regression in §8
(assay +0.012, the smallest coefficient) because that measured an unconditional
additive contribution while this is conditional on same tissue -- the
conditional question is the one the objective asks. **If the goal is "same
tissue, same codes regardless of X", the X to attack is ASSAY.**

### Consequences

1. The "five failed interventions" narrative was an artefact of the metric.
   On the stated objective, niche cross-panel agreement improved 15.9% and
   tissue separation 30%.
2. `nodecay/best` is the best model for this objective, not `ep3` -- the
   opposite of what identification says (§11), so the paper has to pick a
   checkpoint per claim or report both.
3. Neither the adversary nor more MMD is indicated by this. An assay-targeted
   term is.
4. iLISI should be reported as a fraction of its permutation oracle or dropped
   for the quantised embeddings; its nominal ceiling is unreachable (§14).


## 16. Batch conditioning was inert at predict for every run, and it was silent

Evaluating FiLM required fixing predict, and the fix surfaced something larger:
**at predict time, every cell in the corpus was treated as an unseen batch.**
Not some cells, not held-out cells -- all 112,578,039, in all 35 shards:

```
Predict: adata_batch_ids range [0, 0], 1 distinct over 3261663 cells;
         unseen-label cells = 3261663 (these take the mean-embedding fallback).
```

### Root cause: two naming domains

With `batch_key='dataset_batch'` -- required at corpus scale, because
`uns['batch']` collides across datasets (75 distinct values for 636 sections) --
the blob's manifest records a COMPOSITE batch identity,
`<dataset_id>_batch<N>`. Row 342 is `1020_batch5`. The two paths then disagree:

| path | label source | value | lookup |
|---|---|---|---|
| train, `_stamp_batch_ids` | `get_batch_labels()` (manifest) | `1020_batch5` | matches |
| predict, `build_batch_one_hot_from_obs` | `obs_batch` (`uns['batch']`) | `batch5` | misses, every cell |

`obs_batch` is `[str(uns['batch'])] * n_cells`, the BARE value. Densified
against a composite-keyed map, every lookup misses, so every cell gets
`unknown_label_dense_id=0` and `unseen_mask=True`. Consequences:

- the decoder covariate collapses to the MEAN embedding for every cell;
- FiLM receives a one-hot that is constant (index 0) for every cell, so
  encoder conditioning modulates by a constant -- i.e. does nothing.

Training is unaffected: it stamps ids from the manifest identities, which is
why all 416 embedding rows fill and FiLM's columns fill progressively (§10, and
the visitation table below).

### Why nothing ever raised

This is the same class as §12 -- a wrong answer that looks like a working run.
An all-unknown result is indistinguishable from a shard of genuinely novel
batches, which is a legitimate zero-shot case the mean-embedding fallback
exists to serve. Worse, it MASKED a second defect: predict also used the
blob-wide 635-label map instead of the train-restricted 416 one, which should
have produced ids up to 634 against a 416-row embedding and tripped both the
range guard in `VQNiche_Dual.forward` and `nn.Embedding`'s own bounds check.
It never did, across 313 predicts, because the ids never got past 0. Two
independent bugs, the first hiding the second.

Both are now fixed, and both fail loudly if reintroduced:
`initialize_databatch` takes `section_batch_labels` (the authoritative
per-section identities) and raises when a non-empty map matches nothing;
`_predict_label_map` verifies the chosen map's width against the widths stored
in the checkpoint and refuses to start on a mismatch.

### `best` is the wrong arm for any encoder-conditioned run

FiLM's `param_generator` is initialised to zero (`init_mode: identity`), so an
exactly-zero column has never received a gradient. Across the FiLM run:

| checkpoint | FiLM zero-columns | batch_embedding zero-rows |
|---|---|---|
| step 20,000 | 364/416 | 0/416 |
| step 40,000 (lowest val_loss, = `best`) | 309/416 | 0/416 |
| step 100,000 | 180/416 | 0/416 |
| step 200,000 (= `last`) | **0/416** | 0/416 |

At 1.07 section-visits per section per epoch (§13), `best` lands 20% into the
epoch with 74% of the conditioning inert. Scoring it would return a null result
for a reason that has nothing to do with FiLM -- the same non-answer tier2a
gave when weight decay had erased 355/416 columns (§10). Only `last` has all
416 columns live. `batch_embedding` never shows zero rows because it is
randomly initialised, so non-zero there does NOT mean trained; the FiLM
zero-column count is the only unambiguous visitation marker available.

### Consequences

1. **Every reconstruction number in this document was produced with a neutral
   decoder covariate.** They are self-consistent (all runs, all arms, same
   fallback) and identification is untouched -- codes come from the encoder,
   which for non-FiLM runs sees no batch information at all. But they do not
   measure what "batch-conditioned reconstruction" was supposed to mean.
2. **§15's integration table stands.** It reads `Indices_*` from runs with
   `encoder_batch_condition_dim=0`, where batch identity reaches only the
   decoder. Code assignment is unaffected by the covariate fallback.
3. **FiLM has still never been measured.** The 35 shards completed and wrote
   output, but with a constant condition, so `_codeagree.py` on them would
   report the null result for the wrong reason. Re-run predict after the fix
   before drawing any conclusion about encoder conditioning.

## 17. When lus26 degrades, jobs BLOCK rather than fail, and the log is empty

A companion to §12: another infrastructure failure that presents as a code bug.

The test job died on `TERM_RUNLIMIT` after 60 minutes having used **6.00
seconds of CPU** at 43 MB peak, with nothing in its log but the code check. It
was not running slowly -- it was blocked in `torch/__init__.py` loading its C
extension. `/lustre/scratch126` (lus26) had degraded, and both the venv and the
corpus blob live there:

| path | read throughput | how measured |
|---|---|---|
| NFS shared venv (`/nfs/team361/sb75/.venvs/squint`) | 391 MB/s | different filesystem, healthy |
| lus26 venv (`squint/.venv`) | 15.6 MB/s, later 343 kB/s | O_DIRECT / uncached |
| lus26 blob (`DATASETS/gold/...`) | 6.8-11 MB/s | O_DIRECT / uncached |
| same files via cached `dd` | 3.8-4.0 GB/s | **misleading, do not use** |
| `import torch` | > 300 s (normally seconds) | ground truth |

The NFS venv is not a workaround: it fixes the torch import but the corpus blob
exists only on lus26, so an eval is starved either way.

`libtorch_cpu.so` is 475 MB and `libtorch_cuda.so` 855 MB, so at single-digit
MB/s importing torch is many minutes of pure I/O before any user code runs.
Confirmed twice over: `faulthandler.dump_traceback_later` puts the stack in
`create_module` under `from torch._C import *`, and sampling
`/proc/<pid>/wchan` on the blocked process returns **`cl_sync_io_wait`** -- the
Lustre client's synchronous I/O wait, page-faulting the mmap'd library. That
wchan value is the unambiguous marker; reach for it before theorising.

### Measuring it: two traps

**Plain `dd` measures the page cache, not the filesystem.** The first version
of `_iogate.sh` read the head of each file and reported 3.8 GB/s while jobs
were still hanging -- those pages were resident from an earlier probe. Reading
uncached offsets in the SAME file at that moment gave 1.3-4.3 MB/s, and
`iflag=direct` on the *cached* region gave 3.3 MB/s. **Always use
`iflag=direct`.**

**Health is per-file (per-OST), not global.** Observed within the same second:
449 MB/s on the venv library, 6.8 MB/s on a blob section. Gating on the venv
alone would green-light an eval whose data is starved, so check both and take
the worse. Rates also swing violently -- the same venv file went 449 MB/s to
343 kB/s within minutes -- so run the gate immediately before submitting, and
accept that a job can still hit a bad patch.

Diagnosis by per-file collection is unambiguous -- of 17 test files, the 12
that import torch or vqniche ALL hang and the 5 pure-python ones ALL collect
fine, with no exceptions. So a hang in a file you just edited says nothing
about your edit.

### The signature, and the gate

**Six seconds of CPU against an hour of wall clock is the tell.** A real hang
burns CPU; an I/O stall does not. Check `CPU time` against `Run time` in the
LSF summary before reading anything into an empty log.

`_iogate.sh` (untracked) probes venv and blob with `iflag=direct`, requires
>= 30 MB/s on BOTH, and only then times `import torch`. It probes 8 MB rather
than 32: at 343 kB/s a 32 MB probe exceeds its own timeout and reports 0 MB/s,
which reads as a broken probe instead of telling you how bad things are.

Job scripts should carry an I/O preflight that ABORTS with a clear message
rather than letting a stall consume the wall clock -- `_tests_job.sh` now does
(and its `-W` went 60 -> 240).

## 18. Encoder conditioning works: FiLM + nodecay is the best model on the objective

With the predict-time batch identity fixed (§16), FiLM was measurable for the
first time. It is the best run on every column of the stated objective -- same
tissue, same codes regardless of panel.

| run | same panel | **diff panel** | TRANSFER | SEPARATION | xfer assay |
|---|---|---|---|---|---|
| baseline/best | 0.5119 | 0.3114 | 0.6084 | 2.9030 | 0.3424 |
| baseline/last | 0.5341 | 0.3429 | 0.6421 | 2.6486 | **0.4037** |
| tier1/last | 0.4704 | 0.3075 | 0.6538 | 3.7806 | 0.3586 |
| bigblocks/best | 0.4812 | 0.3252 | 0.6759 | 3.1832 | 0.3731 |
| nodecay/best | 0.5235 | 0.3609 | 0.6895 | 3.7704 | 0.3110 |
| ep3/best | 0.4484 | 0.3318 | 0.7400 | 3.7526 | 0.3845 |
| **FiLM+nodecay/last** | 0.4948 | **0.3684** | **0.7446** | **4.6015** | 0.3153 |

Against the arm-matched reference (`tier1/last`: same 200,000 steps, same
`last` arm, no nodecay, no FiLM) it improves *everything*, raw agreement
included: diff-panel +19.8%, transfer +13.9%, separation +21.7%, same-tissue
+11.4%. Versus `baseline/last`: diff-panel +7.4%, separation +73.7%.

### Two ways this could have been spurious, both excluded

**Denominator collapse.** SEPARATION is same-tissue / diff-tissue, so it rises
if a model merely stops agreeing anywhere. FiLM keeps same-tissue agreement
high (0.4281, second only to `nodecay/best`'s 0.4377) while pushing
diff-tissue to the lowest of any run (0.0930 vs baseline 0.1399). Both sides
move the right way. Contrast `tier1/last`, whose separation gain came partly
from same-tissue DROPPING to 0.3844.

**Codebook collapse.** Agreement is trivial if everything piles into a few
codes. It is the reverse -- on the same shard, same 3,051,382 cells:

| run | niche codes used | entropy | top-1 share |
|---|---|---|---|
| FiLM+nodecay/last | **2348**/2700 | 6.856 bits | 0.089 |
| nodecay/best | 877/2700 | 4.715 bits | 0.246 |

FiLM achieves higher cross-panel agreement while using 2.7x more of the
codebook at 2.1 more bits of entropy.

### What is not yet isolated

`nodecay/best` is step 40,000 and FiLM is step 200,000, so the FiLM-vs-nodecay
comparison confounds conditioning with 5x the training. **Resolved in section
19** by scoring `tier1-nodecay` at arm=last: FiLM's contribution is larger
than this section's arm-matched comparison suggested, not smaller.

### Assay is still the nuisance, and interventions are making it worse

FiLM did nothing for assay transfer (0.3153, effectively tied with
`nodecay/best`'s 0.3110) and `baseline/last` remains the BEST on it (0.4037).
Every intervention has improved panel transfer while degrading assay transfer.
Consistent with §15: assay is the dominant nuisance and needs a targeted term.

The signal for one is real but thin: 10 of 20 tissues span more than one assay
(492/636 sections), but only **2,014 of 28,043 same-tissue pairs (7.2%)** are
cross-assay, because Xenium is 560/636 sections. 83% of those pairs sit in
liver, skin, lung and ovary; skin's minority is 2 sections and breast,
pancreas, heart and prostate have 1 each. An assay term would be driven by a
handful of sections, so it risks fitting them rather than the effect.

## 19. FiLM isolated: +29.5% on the objective, and it is not a coarse-codebook artefact

`nodecay/last` (step 200,000, arm `last`) closes the confound left open in
§18: same 200,000 steps, same no-decay exemption, same arm.

**It is not a single-variable comparison, though.** Diffing the saved configs
and confirming against the checkpoints, `FiLM+nodecay` changes TWO things
relative to `nodecay`:

  - encoder FiLM conditioning on `cell_batch_id` (identity init, bias, no
    residual) -- `conditioning_module.param_generator` is (512, 416);
  - `decoder_covariate_embed_dim` 16 -> 64, i.e. `batch_embedding` goes from
    (416, 16) to (416, 64).

Both hand the model more batch capacity, one on the encoder side and one on the
decoder side, so the delta below belongs to the BUNDLE. Attributing it to FiLM
alone would need a third run: nodecay + `decoder_covariate_embed_dim=64` and no
FiLM. Everything else about the comparison is matched:

| | nodecay/last | FiLM+nodecay/last | delta |
|---|---|---|---|
| **diff panel** (the objective) | 0.2845 | 0.3684 | **+29.5%** |
| same tissue | 0.3594 | 0.4281 | +19.1% |
| separation | 3.6696 | 4.6015 | +25.4% |
| transfer | 0.6418 | 0.7446 | +16.0% |

All four improve. The earlier reading -- "+2.1% over nodecay, so most of the
gain is the no-decay fix rather than FiLM" -- was an artefact of comparing
against `nodecay/best` at step 40,000. That checkpoint is not a neutral
reference: within the same run, step 40,000 scores 0.3609 and step 200,000
scores 0.2845, so `best` was the most FAVOURABLE point available to nodecay.

### Codebook granularity, and why two rows of the table are not comparable

The metric is 1-JS between per-section distributions over 2,700 composite
codes. Fewer codes in use means lower-dimensional distributions and
mechanically higher similarity, so agreement must be read against how finely
each run quantises. Measured on the same shard (3,051,382 cells), niche branch:

| run | codes used | entropy | effective codes (2^H) | top-1 | diff panel |
|---|---|---|---|---|---|
| baseline/best | 785 | 4.654 | 25.2 | 0.291 | 0.3114 |
| baseline/last | 1528 | 5.620 | 49.2 | 0.378 | 0.3429 |
| nodecay/best | 877 | 4.715 | 26.3 | 0.246 | 0.3609 |
| bigblocks/best | 2547 | 6.298 | 78.7 | 0.269 | 0.3252 |
| nodecay/last | 2447 | 6.683 | 102.7 | 0.120 | 0.2845 |
| ep3/best | 2388 | 6.833 | 114.0 | 0.153 | 0.3318 |
| **FiLM+nodecay/last** | 2348 | 6.856 | **115.9** | 0.089 | **0.3684** |
| tier1/last | 2404 | 7.305 | 158.2 | 0.044 | 0.3075 |

`nodecay/best` reaches 0.3609 on **26 effective codes** -- a quarter of the
granularity of the fine-grained runs, and `baseline/last` puts 37.8% of all
cells in a single code. Those rows are coarse-quantisation artefacts and should
not be compared with the rest.

Among runs at comparable granularity (~100-160 effective codes) the ranking is
unambiguous, and FiLM wins on the objective at essentially the same granularity
as its two nearest rivals:

    FiLM+nodecay/last  115.9 eff -> 0.3684
    ep3/best           114.0 eff -> 0.3318
    nodecay/last       102.7 eff -> 0.2845
    tier1/last         158.2 eff -> 0.3075

Granularity is not the whole story -- `tier1/last` has the MOST effective codes
and still scores low, and the correlation across all eight runs is only
r = -0.25 -- so agreement does carry real signal. But it is enough to
disqualify the two coarse rows, and it removes the last way FiLM's win could
have been mechanical: FiLM has the highest agreement while using slightly MORE
effective codes than `nodecay/last` (115.9 vs 102.7) and a far flatter
distribution (top-1 0.089 vs 0.120).

### Consequence

`FiLM+nodecay/last` is the model to report for the integration claim -- as a
bundle (encoder conditioning + a 4x wider decoder covariate), which is what
was actually run. Note it
is NOT the best on identification -- that is still `ep3` (§11) -- so the paper
reports a checkpoint per claim, as §15 already anticipated.

## 20. Can the corpus support an assay-targeted term? Only with stratified blocks

§15 and §19 both point at assay as the dominant nuisance, so the question is
whether the data can actually train a term against it. The binding constraint
is not how many cross-assay pairs exist -- it is how often one lands in a
single training BLOCK, because that is when an MMD or adversary scoped by
tissue and targeted at assay can produce a gradient at all.

### The signal that exists (TRAIN sections only -- held-out ones cannot train)

| tissue | cross-assay pairs | share | minority sections | minority cells | composition |
|---|---|---|---|---|---|
| liver | 788 | 50.4% | **22** | 3,918,830 | cosmx 2, merfish 20, xenium 34 |
| ovary | 272 | 17.4% | 4 | 881,623 | merfish 4, xenium 68 |
| brain | 165 | 10.5% | 2 | 312,563 | cosmx 1, merfish 1, xenium 82 |
| colon | 117 | 7.5% | **11** | 4,682,438 | cosmx 9, merfish 2, xenium 9 |
| lung | 108 | 6.9% | 4 | 1,320,223 | cosmx 2, merfish 2, xenium 26 |
| skin | 100 | 6.4% | 2 | 644,736 | merfish 2, xenium 50 |
| breast / heart / prostate | 9 / 3 / 2 | 0.9% | 1 each | | |

1,564 pairs total. **Pair count is the wrong measure of diversity.** Brain has
165 pairs from exactly TWO minority sections (one cosmx, one merfish) against
82 xenium -- the same two sections re-paired 165 times. Effective diversity is
the minority-section count, and by that measure only **liver (22) and colon
(11)** have any depth; everything else is 4 sections or fewer.

This also corrects the earlier note in §18, which named liver/skin/lung/ovary
as holding 83%. That was computed over all 636 sections; restricted to train it
is liver/ovary/brain/colon at 86%, and by effective diversity it is really just
liver and colon.

### Why the current configuration cannot use it

Blocks hold `sections_per_block: 8` capped by `max_cells_per_block: 900,000`,
and the cap binds -- 8 average sections is ~1.84M cells -- giving K = 3.9
sections per block, which matches the 1.07 section-visits of §13.

    P(a given section pair co-occurs)  = C(3.9,2)/C(417,2) = 6.9e-05
    E[cross-assay same-tissue pairs per block] = 1,564 x 6.9e-05 = 0.108
    -> ~11% of blocks carry any assay signal

A 200,000-step run at batch 512 over 900,000-cell blocks consumes ~114 blocks,
so the term would be active for ~11% of steps but would see only **~12 distinct
cross-assay block compositions in the entire run**, half of them liver. That is
enough to fit those sections and not enough to learn assay invariance.

### What would fix it

| scenario | K | P(block has signal) |
|---|---|---|
| current (900k budget) | 3.9 | 10.8% |
| bigblocks budget (1.9M) | 8.0 | 50.5% |
| stratified assembly | 3.9 | ~100% |

Raising the block budget helps because co-occurrence scales as C(K,2): the
existing `bigblocks` config alone takes it from 11% to 50%. But the real fix is
**stratified block assembly** -- deliberately co-scheduling same-tissue,
different-assay sections rather than relying on a random shuffle to collide
them. That reaches ~100% of blocks at no extra memory, and it bounds the
experiment by the 1,564 available pairs instead of by luck.

### Experimental design this implies

An assay term is effectively a liver-and-colon experiment. So the honest test
is to scope the term to those two tissues and measure whether assay transfer
improves on the tissues it never saw (ovary, lung, skin). Measuring it on liver
would not distinguish invariance from memorising 22 sections.

## 21. FiLM does not cost identification, and ep3's lead is an ARM effect

§19 left one thing unknown: whether encoder conditioning buys integration at
the expense of NMI/ARI, the paper's primary claim. It does not.

Held-out TEST split, level-0 codes, unweighted mean over the 81 label keys.
**These are not the §11 headline numbers** -- those come from `_tier0*.py`,
which aggregates per dataset under a different protocol. Everything below is
recomputed the same way for every run so the columns are comparable to each
other:

| run | cell NMI | cell ARI | niche NMI | niche ARI |
|---|---|---|---|---|
| baseline/best | 0.3803 | 0.2159 | 0.2945 | 0.1434 |
| baseline/last | 0.2875 | 0.1120 | 0.2502 | 0.0981 |
| tier1/best | 0.3545 | 0.1885 | 0.3271 | 0.1421 |
| tier1/last | 0.3367 | 0.1540 | 0.2992 | 0.1204 |
| nodecay/best | 0.3818 | 0.1999 | 0.3117 | 0.1273 |
| nodecay/last | 0.3207 | 0.1293 | 0.2862 | 0.1027 |
| **ep3** | **0.3884** | **0.2056** | 0.3190 | **0.1773** |
| FiLM+nodecay/last | 0.3225 | 0.1319 | 0.3003 | 0.1350 |

Against its arm-matched control, `nodecay/last`, FiLM is neutral to positive:
cell NMI +0.6%, cell ARI +1.9%, niche NMI +4.9%, **niche ARI +31.5%**. So the
+29.5% code-agreement gain of §19 is not paid for in identification.

### `best` beats `last` in every single-epoch run

| run | cell NMI best -> last |
|---|---|
| baseline | 0.3803 -> 0.2875 (-24%) |
| tier1 | 0.3545 -> 0.3367 (-5%) |
| nodecay | 0.3818 -> 0.3207 (-16%) |

Identification DEGRADES over a 200,000-step (≈1 epoch) run, and `ep3` -- three
epochs, where `best` and `last` coincide -- is the strongest of all. So
FiLM/last trailing ep3 by 16.9% on cell NMI is largely an arm effect, not a
FiLM effect: FiLM has no usable `best` arm, because at step 40,000 309/416 of
its conditioning columns are still exactly zero (§18).

This is the same non-monotonicity §19 found for code agreement, where nodecay
went 0.3609 at 40k to 0.2845 at 200k. One epoch is a bad place to stop on both
metrics, and it is where five of the seven runs stop.

### Consequence

**FiLM + nodecay + 3 epochs is now the clearly indicated next run.** ep3's
advantage is training length and FiLM's is integration; the two are
independent branches off `nodecay` that have never been combined, and this
section removes the reason to fear the combination -- conditioning costs
nothing on identification.

### Cost note

Pooling took 68 minutes; the metrics step took 8 h 50 m (CPU time 36,102 s vs
run time 35,932 s, so genuinely compute-bound, not the §17 I/O stall). It
scores 4 splits x 4 code keys x 81 label keys ~= 1,300 clusterings over 20.1M
cells, and `compute_inference_metrics.py` has no `--splits` flag, so `all`,
`train` and `validation` are always computed even when only `test` is wanted.
A splits filter would cut roughly 4x off the dominant cost.

## 22. The FiLM gain is an INTERACTION: neither half does anything alone

§19 measured a bundle -- encoder FiLM plus `decoder_covariate_embed_dim`
16 -> 64 -- at +29.5% on cross-panel code agreement and could not say which
half earned it. Two runs complete the square, all four cells at 200,000 steps,
`nodecay`, arm `last`, differing only in these two axes:

| | cov16 | cov64 |
|---|---|---|
| **no FiLM** | 0.2845 (`nodecay/last`) | 0.2908 (**+2.2%**) |
| **FiLM** | 0.2843 (**-0.1%**) | **0.3684** (**+29.5%**) |

**Neither component alone does anything.** FiLM by itself moves the objective
by -0.1%; a 4x wider decoder covariate by itself moves it +2.2%. Their
individual effects sum to +2.1% against +29.5% observed, so roughly 93% of the
gain is interaction.

### It is not a granularity artefact

Agreement rises mechanically as fewer codes are used (§19), so the obvious
worry is that the winning cell simply quantises coarsely. It does not -- the
check comes out backwards:

| run | codes | entropy | effective (2^H) | top-1 | diff panel |
|---|---|---|---|---|---|
| nodecay/last (cov16, -) | 2447 | 6.683 | 102.7 | 0.120 | 0.2845 |
| cov64-only (cov64, -) | 2381 | 6.510 | 91.1 | 0.130 | 0.2908 |
| FiLM-only (cov16, F) | 2331 | 6.855 | **115.7** | 0.116 | 0.2843 |
| FiLM+nodecay (cov64, F) | 2348 | 6.856 | **115.9** | 0.089 | **0.3684** |

`FiLM-only` and `FiLM+nodecay` sit at **the same granularity** -- 115.7 vs
115.9 effective codes, entropies equal to three decimals -- and differ by
29.6% on the objective. Nothing about coarseness can explain that gap; the
only difference is covariate width. Meanwhile `cov64-only` has the COARSEST
codebook of the four, which should flatter it, and still scores 0.2908.

### The mechanism this implies

FiLM alone does change the model -- it raises effective codes 102.7 -> 115.7
and flattens the distribution -- it just does not improve transfer. That is
the conditional-autoencoder division of labour failing to close:

  - the encoder can only strip batch identity out of the codes if the decoder
    can put it back, and 16 dimensions across 416 batches cannot;
  - the decoder can absorb batch at 64 dimensions, but without FiLM the
    encoder has no signal telling it what to strip.

Only both together complete the loop, which is why the effect is
multiplicative rather than additive. SEPARATION behaves the same way, though
less starkly: +13.1% and +2.5% alone, +25.4% together.

### Caveat

One run per cell, no seed replicates. The three null cells span 0.2843-0.2908,
a 2.3% spread, and the winning cell sits 27% above the top of that range --
about 13x the observed noise. Strong for n=1, but a seed replicate of the
winning cell would settle it, and is cheap at 5 h.

### Consequence

**Keep both in the 3-epoch run.** The cheaper cov16 variant is not a viable
economy: at cov16 FiLM is worth nothing. This also retires the §19 caveat that
the bundle was unattributed -- it is attributed, to the pair.

## 23. The replicate: the interaction survives, its size does not, and my noise estimate was wrong in kind

§22 closed with "one run per cell... a seed replicate would settle it, and is
cheap at 5 h." It did settle it, and not in §22's favour.

| cell | diff panel | separation |
|---|---|---|
| nodecay/last (cov16, -) | 0.2845 | 3.6696 |
| cov64-only (cov64, -) | 0.2908 | 4.1497 |
| FiLM-only (cov16, F) | 0.2843 | 3.7618 |
| **FiLM+cov64 seed 0** | **0.3684** | **4.6015** |
| **FiLM+cov64 seed 1** | **0.3159** | 4.0642 |

### What holds

The direction. Both seeds clear every null cell on the objective -- seed 1's
0.3159 is +8.6% above the highest null (0.2908), seed 0's 0.3684 is +26.7%.
The two-seed mean of 0.3422 is **+19.4%** over the null mean.

### What does not

**The effect size.** §22 reported +29.5% from seed 0 alone. Two seeds give
+19.4%, and the seed-to-seed spread on the winning cell is **14.3%** --
comparable to the effect being measured.

**Separation.** Seed 0 led every run at 4.6015 and §22 claimed +25.4%. Seed 1
gives 4.0642, BELOW `cov64-only`'s 4.1497. The separation claim is not
supported by two seeds and should be dropped.

**And the reasoning behind §22's confidence was invalid.** §22 argued the
effect was "about 13x the observed noise", taking the 2.3% spread across the
three null cells as the noise floor. Those three cells are DIFFERENT
CONFIGURATIONS, not replicates; their agreement measures how little the two
components do alone, and says nothing about run-to-run variance. The single
actual replicate puts seed variance at 14.3% -- 6x that spread -- so the
13x claim had no basis. Tight agreement among different configurations is not
evidence of a small noise floor.

### It is not a training difference

Both seeds are equivalent where it could matter: step 200,000, 0/416 FiLM
columns zero, |FiLM| mean 0.0196 vs 0.0199, batch-embedding mean 0.0613 vs
0.0602, val_loss 1771.042 vs 1775.019 (0.2% apart). The eval is identical too
-- same 64-dim embedding, condition dim 416, same 16,546,102 unseen cells. The
variance is in the metric, not the fit.

### Consequence

With 14.3% seed variance on the winner, one run per null cell cannot size the
interaction. Before this is a paper claim the three null cells need replicates
too -- 3 x 5 h, or at minimum a second seed for `cov64-only` and `FiLM-only`,
the two the interaction argument rests on. What can be stated now is the
direction: encoder conditioning plus a widened decoder covariate beats either
alone, by something around 20% with wide error.

## 24. With two seeds per cell the interaction shrinks, the cells overlap, and three epochs does not help

§23 replicated the winning cell and found 14% seed variance. Replicating the
two ablation cells settles what that means, and it is not favourable.

| cell | seeds | range | mean | spread |
|---|---|---|---|---|
| cov16, no FiLM | 1 | 0.2845 | 0.2845 | -- |
| cov64, no FiLM | 2 | 0.2908 - 0.3055 | 0.2982 | 5.1% |
| cov16, FiLM | 2 | 0.2843 - **0.3220** | 0.3031 | 13.3% |
| cov64, FiLM | 2 | **0.3159** - 0.3684 | 0.3422 | 16.6% |
| cov64, FiLM, 3 epochs | 1 | 0.3028 | 0.3028 | -- |

### The null cells are not tight either

§22 read a 2.3% spread across three differently-configured single runs as a
noise floor. With replicates, `cov16,FiLM` spreads **13.3%** -- nearly the
winner's 16.6%. The 2.3% was an accident of which single draws happened to be
taken, exactly as §23 warned.

### The winning cell now OVERLAPS a null cell

`cov16,FiLM` reaches 0.3220; `cov64,FiLM` falls to 0.3159. A run with FiLM
alone beat a run with both. The winner is still separated from `cov64,noFiLM`
(max 0.3055 < min 0.3159), but on two draws per cell that is one lucky pair
away from collapsing too.

### The interaction is smaller than claimed and no longer super-additive enough to matter

On two-seed means against the cov16/no-FiLM baseline:

    cov64 alone  +4.8%      FiLM alone  +6.6%      sum  +11.4%
    both        +20.3%      (§22 claimed +29.5% from single runs)

Still super-additive -- 20.3% against a predicted 11.4% -- but the margin is
9 points while within-cell spread is 13-17 points. **The interaction is not
resolvable at this sample size.** §22's "~93% of the gain is interaction" was
an artefact of single draws; the honest figure is that roughly half the
combined effect exceeds the sum of its parts, with error bars wider than the
excess.

### Three epochs made the objective WORSE

`cov64,FiLM` at 562,680 steps scores **0.3028**, below both 200,000-step seeds
(0.3159, 0.3684) and level with the `cov16,FiLM` mean. Its diff-tissue is the
highest of any run (0.1043) and its separation the lowest (3.4814). n=1, and
inside the 200k range's spread, so this is not conclusive -- but it is the
third time more training has hurt this metric (§19: nodecay 0.3609 at 40k ->
0.2845 at 200k; §21: identification degrades best -> last in every one-epoch
run). The expectation from §18 that FiLM would improve with more gradients per
conditioning column is not borne out on the objective.

Its val_loss trajectory says the same: 1774, 1792, 1851, 1808, 1888, 1908,
1777, 1835, 1859, 1757 across the ten checkpoints -- a 150-unit band with no
trend, where the final checkpoint is the minimum by luck. Lowest val_loss of
any run, and mid-table on the metric that matters.

### Consequence: this metric cannot resolve 20% effects from single runs

Run-to-run spread is 13-17% on three of the four cells. Every single-run
comparison in §15, §19 and §22 sits inside that band, so their rankings are
provisional and their percentages should not be quoted. What survives is
ordinal and weak: conditioning plus a widened covariate is the best cell on
the mean of two seeds, by a margin the sample size cannot defend.

Before any of this is a paper claim, either the metric needs its variance
reduced (more sections per estimate, or a paired design across seeds) or the
cells need enough seeds to separate ~20% effects against 15% noise -- on the
order of 5 per cell, i.e. ~25 h of GPU per cell at 200,000 steps.
