"""
Predict without expression, and split-stratified post-hoc metrics.

Two changes, both forced by the same arithmetic. The cross-panel builder
allocates a DENSE zero-filled `.X` at the gene vocabulary for every section --
`np.zeros((n_obs, 9574), float32)`. That is 590 GiB over the 219 held-out
sections and 3.9 TiB over all 636, and outside `full` mode nothing ever reads
it: identification, integration, latent geometry and codebook utilisation all
live in `obsm` and `obs`. Dropping it is the difference between 49 sharded
jobs and one.

The metrics side was pooled over every cell in the object. On a corpus run that
object mixes 96M training cells with 16.5M held-out ones, so one iLISI / MMD /
avg-cosine number over the mixture describes neither.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "examples" / "run_squint.py"
METRICS = ROOT / "examples" / "compute_inference_metrics.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"{path.name} not importable here: {exc}")
    return mod


# --------------------------------------------------------------------------- #
# expression-free build
# --------------------------------------------------------------------------- #

def test_the_builder_takes_an_include_expression_switch():
    tree = ast.parse(DRIVER.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_build_clean_adata_from_inference")
    names = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "include_expression" in names
    assert "keep_obs_cols" in names


def test_the_dense_widening_is_skipped_without_expression():
    """
    The allocation this exists to avoid is the `np.zeros((n_obs, len(gene_names)))`
    in the cross-panel branch. It must sit behind the switch.
    """
    src = DRIVER.read_text()
    assert "if include_expression and gene_names is not None:" in src, (
        "the gene_names reindex/widening block is not gated on "
        "include_expression -- the dense zero-fill still runs"
    )
    i = src.index("np.zeros((sub.n_obs, len(gene_names))")
    j = src.index("if include_expression and gene_names is not None:")
    assert j < i, "the dense allocation is reachable before the gate"


def test_predict_turns_expression_off_for_every_non_full_cache_mode():
    src = DRIVER.read_text()
    assert '_include_expr = (_cache_mode == "full")' in src, (
        "expression must be built ONLY in full mode; codes and codes+latents "
        "carry no X_hat layer, so nothing can read .X"
    )
    i = src.index("_include_expr = ")
    j = src.index("include_expression=_include_expr")
    assert i < j


@pytest.mark.parametrize("mode,expect", [
    ("full", True), ("codes", False), ("codes+latents", False),
])
def test_the_mode_to_expression_rule(mode, expect):
    assert (mode == "full") is expect


# --------------------------------------------------------------------------- #
# split-stratified metrics
# --------------------------------------------------------------------------- #

class _FakeAdata:
    def __init__(self, splits, key="terra_split"):
        import pandas as pd
        self.n_obs = len(splits)
        self.obs = pd.DataFrame({key: splits})


def test_split_masks_prefer_terra_split_and_keep_val_distinct():
    m = _load(METRICS, "_cim_under_test")
    a = _FakeAdata(["train"] * 5 + ["validation"] * 3 + ["test"] * 4)
    got = dict(m.resolve_split_masks(a))
    assert set(got) == {"all", "train", "validation", "test"}
    assert got["all"].sum() == 12
    assert got["validation"].sum() == 3
    assert got["test"].sum() == 4, "validation must not be folded into test"


def test_split_masks_fall_back_to_data_split():
    m = _load(METRICS, "_cim_under_test")
    a = _FakeAdata(["train"] * 4 + ["test"] * 2, key="data_split")
    got = dict(m.resolve_split_masks(a))
    assert "data_split=test" in got and got["data_split=test"].sum() == 2


def test_split_masks_degrade_to_all_when_nothing_is_recorded():
    import pandas as pd
    m = _load(METRICS, "_cim_under_test")
    a = _FakeAdata([])
    a.obs = pd.DataFrame(index=range(7))
    a.n_obs = 7
    got = m.resolve_split_masks(a)
    assert [n for n, _ in got] == ["all"]


def test_a_one_cell_split_is_dropped():
    """Every metric here needs at least a pair; several need far more."""
    m = _load(METRICS, "_cim_under_test")
    a = _FakeAdata(["train"] * 6 + ["test"])
    assert "test" not in dict(m.resolve_split_masks(a))


def test_integration_and_cosine_rows_carry_a_split():
    src = METRICS.read_text()
    for metric in ("iLISI", "ASW", "MMD"):
        assert f'{{"split": _split, "emb_key": emb_key, "metric": "{metric}"' in src, metric
    assert '"split": _split, "emb_key": emb_key,\n                     "avg_cosine_similarity"' in src


def test_integration_slices_the_embedding_and_the_batch_vector_together():
    """
    A mask applied to the embedding but not to the batch labels would pair
    cell i's embedding with cell j's batch -- silently, and only on the
    stratified rows.
    """
    src = METRICS.read_text()
    i = src.index("for _split, _mask, emb_key in _jobs:")
    w = src[i:i + 900]
    assert "np.asarray(adata.obsm[emb_key])[_mask]" in w
    assert "batch = _batch_all[_mask]" in w


# --------------------------------------------------------------------------- #
# codebook utilisation
# --------------------------------------------------------------------------- #

def test_utilisation_normalises_each_rvq_level_by_its_own_size():
    """
    The levels are [30, 90]. One scalar denominator would report level 1 as
    3x its true utilisation.
    """
    src = METRICS.read_text()
    i = src.index("=== Codebook utilisation ===")
    w = src[i:i + 3000]
    assert "sizes[lvl]" in w, "per-level size lookup missing"
    assert '"utilisation": (float(vals.size / size) if size else None)' in w


def test_utilisation_reports_perplexity_too():
    """
    Fraction-used saturates: 30/30 looks perfect whether the mass is uniform
    or 99% on one code. exp(entropy) does not.
    """
    src = METRICS.read_text()
    i = src.index("=== Codebook utilisation ===")
    w = src[i:i + 3000]
    assert "perplexity_norm" in w and "np.exp(ent)" in w


def test_perplexity_separates_uniform_from_concentrated_usage():
    """The property the metric is there for, on the numbers themselves."""
    def pplx(counts):
        p = np.asarray(counts, dtype=float); p = p / p.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    uniform = pplx([100] * 30)
    concentrated = pplx([9701] + [1] * 29)
    assert uniform == pytest.approx(30.0, rel=1e-6)
    assert concentrated < 2.0
    # both have utilisation 30/30 = 1.0; only perplexity tells them apart
    assert uniform / concentrated > 15


def test_no_size_means_no_fraction_rather_than_a_guessed_one():
    src = METRICS.read_text()
    i = src.index("=== Codebook utilisation ===")
    w = src[i:i + 3000]
    assert "if size else None" in w, (
        "utilisation must be None when the codebook size is unknown, not "
        "normalised by a guess"
    )


def test_codebook_sizes_survive_the_uns_round_trip():
    """
    `uns['squint']['codebook_sizes']` is written as a list and comes BACK from
    h5ad as a numpy array. `array or []` raises "truth value of an array with
    more than one element is ambiguous" -- which is what happened on the first
    real run, after integration had already completed.
    """
    src = METRICS.read_text()
    assert "list(_sizes.get(_branch) or [])" not in src, (
        "truthiness test on a value that round-trips from uns as an ndarray"
    )
    i = src.index("=== Codebook utilisation ===")
    w = src[i:i + 3000]
    assert "_s = _sizes.get(_branch)" in w
    assert "[] if _s is None else" in w


def test_the_ambiguous_truth_value_pattern_is_gone_everywhere_in_that_block():
    import numpy as np
    # the failing expression, reproduced
    arr = np.array([30, 90])
    with pytest.raises(ValueError, match="ambiguous"):
        _ = arr or []
    # the replacement, on the same value
    assert ([] if arr is None else [int(v) for v in arr]) == [30, 90]
    assert ([] if None is None else "unreachable") == []


# --------------------------------------------------------------------------- #
# --inference-cache-mode has to reach the model
# --------------------------------------------------------------------------- #

def test_predict_stamps_the_cache_mode_onto_the_loaded_model():
    """
    `initialize_model` is what sets `inference_cache_mode` and rebuilds the
    caches (initialize.py:947-952). Predict does NOT call it -- it goes straight
    to `Model.load_from_checkpoint`, so the flag was a no-op and the model kept
    caching at `full`.

    Invisible while every run passed `full`. The first `codes+latents` run
    cached all four gene-width keys anyway and died assigning them to the
    expression-free AnnData:

        ValueError: Value passed for key 'X_hat' is of incorrect shape.
        Value had shape (444251, 9574) while it should have had (444251, 0)

    The silent half was the expensive one: 4 x 37.4 kB/cell of tensors the job
    had asked not to build, which is most of why four sharded elements hit a
    700 GB limit.
    """
    src = DRIVER.read_text()
    i = src.index("model = Model.load_from_checkpoint(")
    w = src[i:i + 1800]
    assert 'model.inference_cache_mode = str(_mode)' in w, (
        "the cache mode is never stamped on the model predict actually uses"
    )
    assert "model._init_inference_data_caches()" in w, (
        "stamping the attribute is not enough -- the caches were already built "
        "by __init__ and must be rebuilt for the mode to take effect"
    )
    assert w.index('model.inference_cache_mode') < w.index("model.eval()")


def test_an_unsupported_model_refuses_rather_than_caching_at_full_width():
    src = DRIVER.read_text()
    i = src.index("model = Model.load_from_checkpoint(")
    w = src[i:i + 1800]
    assert "Refusing rather than" in w, (
        "a model without `_init_inference_data_caches` must raise, not silently "
        "ignore the flag -- that is precisely the failure this fixes"
    )


@pytest.mark.parametrize("mode,drops_gene_width,drops_latents", [
    ("full", False, False),
    ("codes+latents", True, False),
    ("codes", True, True),
])
def test_the_documented_drop_sets(mode, drops_gene_width, drops_latents):
    """
    Verified against the real corpus checkpoint: 12 cache keys ->
    8 for codes+latents (X, X_nbr, X_hat, X_hat_nbr dropped) and
    4 for codes (those plus the four H_* latents).
    """
    GENE_WIDTH = {"X", "X_nbr", "X_hat", "X_hat_nbr"}
    LATENTS = {"H_latent_cell", "H_quantized_cell",
               "H_latent_niche", "H_quantized_niche"}
    drop = set()
    if mode == "codes":
        drop = GENE_WIDTH | LATENTS
    elif mode in ("codes+latents", "codes_latents"):
        drop = GENE_WIDTH
    assert bool(GENE_WIDTH & drop) is drops_gene_width
    assert bool(LATENTS & drop) is drops_latents


def test_gene_width_caches_dominate_the_memory_at_corpus_width():
    """
    Why the no-op was costly rather than merely untidy, at V=9,574.
    """
    V = 9574
    gene_width = 4 * V * 4
    latents = 4 * 256 * 4
    assert gene_width / latents > 30
    # per-shard, at the 3.5M-cell budget
    assert 3.5e6 * gene_width / 2**30 > 480      # GiB of pure waste


# --------------------------------------------------------------------------- #
# the cache WRITER has to respect the pruned key set too
# --------------------------------------------------------------------------- #

MODEL = ROOT / "src" / "vqniche" / "models" / "vqniche_dual.py"


def test_mode_optional_keys_are_written_through_a_guard():
    """
    `_init_inference_data_caches` prunes `self.cache_keys`, and both the
    persistent caches and predict's fresh local dict are built FROM that list.
    `_cache_inference_data` appended to the pruned keys anyway:

        KeyError: 'X'   (vqniche_dual.py:1236, predict_step)

    which only appeared once the mode fix (43b4d0e) made the pruning real --
    every earlier run silently cached at full width. Both halves are needed:
    stamping the mode, and honouring it at the write site.
    """
    src = MODEL.read_text()
    for key in ("X", "X_nbr", "X_hat", "X_hat_nbr", "H_latent_cell",
                "H_quantized_cell", "H_latent_niche", "H_quantized_niche"):
        assert f"cache_dict['{key}'].append" not in src, (
            f"{key} is still appended unconditionally; it raises KeyError in "
            f"any mode that drops it"
        )
        assert f"_put('{key}'," in src, f"{key} is not routed through _put"


def test_the_guard_skips_rather_than_creating_the_key():
    """
    `_put` must not resurrect a key the mode deliberately dropped -- that would
    reinstate the 149.6 kB/cell this exists to avoid, just later.
    """
    src = MODEL.read_text()
    i = src.index("def _put(key, value):")
    body = src[i:i + 220]
    assert "cache_dict.get(key)" in body
    assert "setdefault" not in body and "cache_dict[key] = []" not in body


def test_the_neighbour_gather_is_skipped_when_neither_product_is_cached():
    """
    The aggregation is the most expensive step in this function -- the gather
    materialises [n_edges, n_genes], which is what produced the 89.23 GiB CUDA
    OOM at the corpus vocabulary. In `codes` / `codes+latents` neither output is
    kept, so it should not run at all.
    """
    src = MODEL.read_text()
    assert "_want_nbr = ('X_nbr' in cache_dict) or ('X_hat_nbr' in cache_dict)" in src
    i = src.index("_want_nbr =")
    j = src.index("aggregate_1hop_neighbor_features", i)
    between = src[i:j]
    assert "if not _want_nbr:" in between, (
        "the gather runs before the guard -- the cost is paid regardless"
    )


def _simulate_put(cache_keys, writes):
    """The _put contract, on the real key sets."""
    cache = {k: [] for k in cache_keys}
    for k, v in writes:
        lst = cache.get(k)
        if lst is not None:
            lst.append(v)
    return cache


@pytest.mark.parametrize("mode,expect_kept", [
    ("full", 12), ("codes+latents", 8), ("codes", 4),
])
def test_put_never_raises_for_any_mode(mode, expect_kept):
    ALL = ["X", "X_nbr", "X_hat", "X_hat_nbr",
           "H_latent_cell", "H_quantized_cell",
           "H_latent_niche", "H_quantized_niche",
           "XY_coordinates", "adata_batch_ids",
           "Indices_cell", "Indices_niche"]
    GENE_WIDTH = {"X", "X_nbr", "X_hat", "X_hat_nbr"}
    LATENTS = {"H_latent_cell", "H_quantized_cell",
               "H_latent_niche", "H_quantized_niche"}
    drop = set()
    if mode == "codes":
        drop = GENE_WIDTH | LATENTS
    elif mode == "codes+latents":
        drop = GENE_WIDTH
    keys = [k for k in ALL if k not in drop]
    assert len(keys) == expect_kept
    cache = _simulate_put(keys, [(k, 1) for k in ALL])   # writer attempts all
    assert set(cache) == set(keys), "a dropped key came back"
    assert all(len(v) == 1 for v in cache.values())
