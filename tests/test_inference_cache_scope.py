"""
Predict output scoping.

Four cached keys are GENE-WIDTH -- X, X_nbr, X_hat, X_hat_nbr -- and over
hst_corpus_110m each is 112,578,039 x 9,574 x float32 = 4.31 TB, so an unscoped
predict would try to accumulate ~17 TB. The regimes need different subsets:
identification, integration and query-to-reference read codes only;
reconstruction and imputation need the gene-width matrices and must therefore
run on a subset of sections per rung.
"""
import pytest

GENE_WIDTH = {"X", "X_nbr", "X_hat", "X_hat_nbr"}
LATENTS = {"H_latent_cell", "H_quantized_cell",
           "H_latent_niche", "H_quantized_niche"}
CODES = {"Indices_cell", "Indices_niche"}


class _VQ:
    codebook_size = 30
    num_quantizers = 2
    codebook_sizes = [30, 90]
    separate_codebook_per_head = False
    heads = 1


class _Enc:
    def __init__(self):
        self.vq_cell = _VQ()
        self.vq_niche = _VQ()


class _Stub:
    """Enough of VQNiche_Dual for `_init_inference_data_caches` to run."""

    def __init__(self, mode=None):
        self.encoder = _Enc()
        if mode is not None:
            self.inference_cache_mode = mode


def _keys(mode):
    from vqniche.models.vqniche_dual import VQNiche_Dual
    s = _Stub(mode)
    VQNiche_Dual._init_inference_data_caches(s)
    return set(s.cache_keys), s


def test_full_is_the_default_and_keeps_everything():
    """Nothing that worked before may change silently."""
    keys, _ = _keys(None)
    assert GENE_WIDTH <= keys and LATENTS <= keys and CODES <= keys


def test_full_explicit_matches_the_default():
    assert _keys("full")[0] == _keys(None)[0]


def test_codes_drops_gene_width_and_latents():
    keys, _ = _keys("codes")
    assert not (GENE_WIDTH & keys), f"gene-width keys survived: {GENE_WIDTH & keys}"
    assert not (LATENTS & keys)
    # what identification / integration / query-to-reference need must remain
    assert CODES <= keys
    assert "XY_coordinates" in keys and "adata_batch_ids" in keys


def test_codes_latents_keeps_embeddings_but_not_gene_width():
    keys, _ = _keys("codes+latents")
    assert not (GENE_WIDTH & keys)
    assert LATENTS <= keys and CODES <= keys


def test_underscore_spelling_accepted():
    assert _keys("codes_latents")[0] == _keys("codes+latents")[0]


def test_an_unknown_mode_raises():
    with pytest.raises(ValueError, match="must be 'codes'"):
        _keys("indices-only")


def test_the_caches_actually_have_the_scoped_keys():
    """The gate must shape the per-split caches, not just `cache_keys`."""
    keys, s = _keys("codes")
    for split in ("train_inference_data_cache", "val_inference_data_cache",
                  "test_inference_data_cache"):
        cache = getattr(s, split)
        assert not (GENE_WIDTH & set(cache)), split
        assert CODES <= set(cache), split


def test_scoping_is_reported():
    """A silent drop of output keys would be the wrong kind of surprise."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        _keys("codes")
    out = buf.getvalue()
    assert "codes" in out and "dropping" in out


@pytest.mark.parametrize("mode,expected_tb", [
    ("full", 17.2),          # 4 gene-width matrices
    ("codes+latents", 0.0),  # no gene-width
    ("codes", 0.0),
])
def test_the_arithmetic_this_exists_for(mode, expected_tb):
    """Documents the number that motivates the gate, at corpus scale."""
    cells, genes = 112_578_039, 9_574
    keys, _ = _keys(mode)
    tb = len(GENE_WIDTH & keys) * cells * genes * 4 / 1e12
    assert round(tb, 1) == expected_tb
