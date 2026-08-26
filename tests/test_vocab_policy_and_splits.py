"""
Vocabulary policy, section identity, and section-restricted loaders.

Three changes are covered here, all of which exist because names that were
unique at 39 sections are not unique at 636:

  * `min_panels_per_gene` / an explicit `gene_vocab` -- cap the vocabulary and
    slice each section's stored columns to match, REPORTING the loss;
  * `batch_key='dataset_batch'` -- `uns['batch']` is a within-dataset counter,
    so at corpus scale one label names up to 80 unrelated experiments;
  * `section_rows` / `split_sections` -- whole-section splits, so a val pass
    visits only the val sections instead of rebuilding the whole corpus.

The load-bearing assertion throughout is that `gene_ids` keeps indexing the
STORED columns exactly. If a filter and a slice ever disagree, counts land under
the wrong gene names and nothing raises.
"""
import numpy as np
import pytest
import torch

from vqniche.dataset.on_disk_dataset import (
    KSectionBlockLoader,
    OnDiskDatasetBlob,
    OnDiskStreamingDataModule,
    _rel_section_name,
    _section_batch_identity,
)
from vqniche.dataset.transforms import SetExperimentDataKeys

GRAPH_KWARGS = {
    "graph_type": "spatial",
    "coord_type": "generic",
    "spatial_key": "spatial",
    "n_neighs_list": [4],
    "radius_list": None,
    "include_self_loop": True,
    "k": {},
}
EDGE = "spatial_n_neighs_4"


def _write(root, panels, name="xp", seed=0, dataset_ids=None, batches=None):
    """`panels`: section index -> gene names. Counts are the gene's position+1
    so a mis-slice moves a recognisable value."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    silver = root / "silver" / name
    silver.mkdir(parents=True, exist_ok=True)
    for i, genes in panels.items():
        n = 12 + i
        counts = np.tile(
            np.arange(1, len(genes) + 1, dtype="float32"), (n, 1))
        a = ad.AnnData(counts)
        a.var.index = list(genes)
        a.obs["cell_type"] = rng.choice(["A", "B"], size=n)
        a.obs["cell_id"] = [f"b{i}_c{j}" for j in range(n)]
        a.obsm["spatial"] = rng.random((n, 2)) * 100
        a.uns.update(
            batch=(batches or {}).get(i, f"batch{i}"),
            dataset_id=(dataset_ids or {}).get(i, name),
            tissue="syn", species="syn",
        )
        a.write_h5ad(silver / f"section_{i}.h5ad")
    return name


def _build(root, name, **kw):
    return OnDiskDatasetBlob(
        name=name,
        feature_names=["cell_gene_counts"],
        label_names=["cell_types=cell_type"],
        graph_kwargs=GRAPH_KWARGS,
        data_directory_path=root,
        pre_filter=None,
        overwrite=True,
        software_paths={"deepwalk": "", "gosh": ""},
        **kw,
    )


def _transform():
    return SetExperimentDataKeys(
        feature_names=["X"], label_name="cell_types", edge_index_name=EDGE,
    )


def _loader(blob, K=2, transform=None, num_neighbors=(2,), **kw):
    return KSectionBlockLoader(
        dataset=blob, edge_index_name=EDGE, sections_per_block=K,
        batch_size=4, num_neighbors=list(num_neighbors), shuffle=False,
        section_transform=transform,
        batch_label_to_dense=blob.batch_label_to_dense(), **kw,
    )


def _names_of(blob, row):
    """Gene names of a stored section's columns, via its ids."""
    sec = blob.get(row)
    ids = sec.gene_ids.view(-1).tolist()
    return [blob.gene_vocab[i] for i in ids]


# `x` is exclusive to section 0; `y` exclusive to section 2. Everything else is
# corroborated by >= 2 distinct panels.
PANELS = {
    0: ["a", "b", "c", "x"],
    1: ["a", "b", "c"],
    2: ["a", "b", "y"],
}


# --------------------------------------------------------------------------- #
# the equivalence guard: the default must not move
# --------------------------------------------------------------------------- #

def test_threshold_one_is_identical_to_no_threshold(tmp_path):
    """`min_panels_per_gene=1` is the shipped behaviour, bit for bit."""
    a = _build(_write(tmp_path / "a", PANELS) and tmp_path / "a", "xp",
               cross_panel=True)
    b = _build(_write(tmp_path / "b", PANELS) and tmp_path / "b", "xp",
               cross_panel=True, min_panels_per_gene=1)
    assert list(a.gene_vocab) == list(b.gene_vocab)
    for r in range(len(a)):
        sa, sb = a.get(r), b.get(r)
        assert torch.equal(sa.gene_ids, sb.gene_ids)
        assert torch.equal(sa["x_cell_gene_counts"], sb["x_cell_gene_counts"])


def test_threshold_one_keeps_the_full_union(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    assert list(blob.gene_vocab) == ["a", "b", "c", "x", "y"]
    assert blob._gene_cols_per_batch == {}


# --------------------------------------------------------------------------- #
# min_panels_per_gene
# --------------------------------------------------------------------------- #

def test_threshold_two_drops_single_panel_genes(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp",
                  cross_panel=True, min_panels_per_gene=2)
    # 'x' and 'y' each appear in exactly one panel.
    assert list(blob.gene_vocab) == ["a", "b", "c"]


def test_stored_columns_are_sliced_to_the_vocabulary(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp",
                  cross_panel=True, min_panels_per_gene=2)
    # Section 0 loses 'x', section 2 loses 'y', section 1 loses nothing.
    assert _names_of(blob, 0) == ["a", "b", "c"]
    assert _names_of(blob, 1) == ["a", "b", "c"]
    assert _names_of(blob, 2) == ["a", "b"]
    for r in range(len(blob)):
        sec = blob.get(r)
        assert sec["x_cell_gene_counts"].shape[1] == sec.gene_ids.numel()


def test_the_surviving_counts_are_the_right_ones(tmp_path):
    """Counts are position+1, so slicing the wrong columns is visible."""
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp",
                  cross_panel=True, min_panels_per_gene=2)
    # section 0 native ["a","b","c","x"] -> values 1,2,3,4; keep a,b,c = 1,2,3
    assert blob.get(0)["x_cell_gene_counts"][0].tolist() == [1.0, 2.0, 3.0]
    # section 2 native ["a","b","y"] -> 1,2,3; keep a,b = 1,2 (NOT 2,3)
    assert blob.get(2)["x_cell_gene_counts"][0].tolist() == [1.0, 2.0]


def test_drops_are_reported_and_land_in_the_manifest(tmp_path, capsys):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp",
                  cross_panel=True, min_panels_per_gene=2)
    out = capsys.readouterr().out
    assert "Vocabulary cap" in out and "of counts" in out

    import json
    man = json.load(open(blob.processed_dir + "/manifest.json")) if isinstance(
        blob.processed_dir, str) else json.load(
        open(blob.processed_paths[1]))
    assert man["manifest_version"] == 4
    assert man["min_panels_per_gene"] == 2
    drops = {d["rel"]: d for d in man["vocab_drops"]}
    assert len(drops) == 2                      # sections 0 and 2 only
    d0 = next(v for k, v in drops.items() if k.endswith("section_0.h5ad"))
    assert (d0["native_cols"], d0["kept_cols"], d0["dropped_cols"]) == (4, 3, 1)
    # 'x' carried value 4 of the row total 1+2+3+4 = 10
    assert d0["dropped_frac_counts"] == pytest.approx(0.4)


def test_threshold_that_empties_the_vocabulary_raises(tmp_path):
    with pytest.raises(ValueError, match="EMPTY vocabulary"):
        _build(_write(tmp_path, PANELS) and tmp_path, "xp",
               cross_panel=True, min_panels_per_gene=9)


def test_threshold_below_one_raises(tmp_path):
    with pytest.raises(ValueError, match="must be >= 1"):
        _build(tmp_path, "xp", cross_panel=True, min_panels_per_gene=0)


# --------------------------------------------------------------------------- #
# the shared vocabulary — the reason Part E can build two comparable blobs
# --------------------------------------------------------------------------- #

def test_override_is_shared_across_different_section_sets(tmp_path):
    """
    Two blobs over DIFFERENT sections must get identical `gene_ids` for the
    same gene, or checkpoints trained on them have non-comparable V-wide rows.
    """
    _write(tmp_path / "full", PANELS)
    _write(tmp_path / "sub", PANELS)
    full = _build(tmp_path / "full", "xp", cross_panel=True,
                  min_panels_per_gene=2)
    shared = list(full.gene_vocab)
    sub = _build(tmp_path / "sub", "xp", cross_panel=True,
                 gene_vocab=shared,
                 exclude_sections=["section_0.h5ad"])
    assert list(sub.gene_vocab) == shared
    assert len(sub) == 2
    # 'a' is vocabulary id 0 in both, and both index their own columns by it.
    for blob in (full, sub):
        assert blob.gene_vocab.get_loc("a") == 0
    assert _names_of(sub, 0) == ["a", "b", "c"]


def test_override_reports_genes_no_section_carries(tmp_path, capsys):
    _write(tmp_path, PANELS)
    _build(tmp_path, "xp", cross_panel=True, gene_vocab=["a", "b", "zzz"])
    out = capsys.readouterr().out
    assert "appear in NO built section" in out and "zzz" in out


def test_override_without_cross_panel_raises(tmp_path):
    with pytest.raises(ValueError, match="cross_panel=False"):
        _build(tmp_path, "xp", gene_vocab=["a"])


# --------------------------------------------------------------------------- #
# section identity — exclude_sections and the batch key
# --------------------------------------------------------------------------- #

def test_exclude_by_rel_path(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp",
                  cross_panel=True,
                  exclude_sections=["section_1.h5ad"])
    assert len(blob) == 2
    assert all("section_1" not in r for r in blob.section_rels)


def test_ambiguous_stem_raises_instead_of_dropping_several(tmp_path):
    """The corpus has 40 stems covering 162 files; a silent multi-match here
    would mis-scope a whole train/holdout split."""
    _write(tmp_path, {0: ["a", "b"]}, name="ds1")
    _write(tmp_path, {0: ["a", "b"]}, name="ds2")
    with pytest.raises(ValueError, match="more than one section"):
        OnDiskDatasetBlob(
            name="", feature_names=["cell_gene_counts"],
            label_names=["cell_types=cell_type"], graph_kwargs=GRAPH_KWARGS,
            data_directory_path=tmp_path, pre_filter=None, overwrite=True,
            software_paths={"deepwalk": "", "gosh": ""},
            cross_panel=True, exclude_sections=["section_0"],
        )


def test_unmatched_exclusion_still_raises(tmp_path):
    with pytest.raises(ValueError, match="not in"):
        _build(_write(tmp_path, PANELS) and tmp_path, "xp",
               cross_panel=True, exclude_sections=["xp/nope.h5ad"])


def test_composite_batch_key_separates_colliding_labels(tmp_path):
    """
    Two datasets both using 'batch0' must NOT collapse into one batch — the
    corpus case, where `batch0` names 80 unrelated experiments.
    """
    _write(tmp_path, {0: ["a", "b"]}, name="ds", dataset_ids={0: "11"},
           batches={0: "batch0"})
    _write(tmp_path, {1: ["a", "b"]}, name="ds", dataset_ids={1: "22"},
           batches={1: "batch0"})
    legacy = _build(tmp_path, "ds", cross_panel=True)
    assert len(set(legacy.batch_labels)) == 1          # the bug
    assert len(legacy.batch_label_to_dense()) == 1

    fixed = _build(tmp_path, "ds", cross_panel=True, batch_key="dataset_batch")
    assert sorted(fixed.batch_labels) == ["11_batch0", "22_batch0"]
    assert len(fixed.batch_label_to_dense()) == 2


def test_collision_warning_fires(tmp_path, capsys):
    _write(tmp_path, {0: ["a", "b"]}, name="ds", dataset_ids={0: "11"},
           batches={0: "batch0"})
    _write(tmp_path, {1: ["a", "b"]}, name="ds", dataset_ids={1: "22"},
           batches={1: "batch0"})
    _build(tmp_path, "ds", cross_panel=True)
    out = capsys.readouterr().out
    assert "batch label collision" in out


def test_batch_identity_falls_back_without_dataset_id(tmp_path):
    from torch_geometric.data import Data
    d = Data()
    d.obs_batch = ["batch3"]
    d.adata_batch_id = torch.tensor(3)
    assert _section_batch_identity(d, "dataset_batch") == "batch3"
    d.dataset_id = "77"
    assert _section_batch_identity(d, "dataset_batch") == "77_batch3"
    assert _section_batch_identity(d, "batch") == "batch3"


def test_bad_batch_key_raises(tmp_path):
    with pytest.raises(ValueError, match="batch_key must be"):
        _build(tmp_path, "xp", batch_key="nonsense")


def test_rel_section_name_is_subdir_qualified(tmp_path):
    p = tmp_path / "silver" / "ds1" / "adata_batch0.h5ad"
    assert _rel_section_name(p, tmp_path / "silver") == "ds1/adata_batch0.h5ad"


# --------------------------------------------------------------------------- #
# section-restricted loaders
# --------------------------------------------------------------------------- #

def test_section_rows_for_resolves_names(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    assert blob.section_rows_for(["section_2.h5ad"]) == [2]
    assert blob.section_rows_for(["section_0", "section_2"]) == [0, 2]
    with pytest.raises(KeyError):
        blob.section_rows_for(["xp/absent.h5ad"])


def test_restricted_loader_visits_only_those_sections(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    ld = _loader(blob, K=1, transform=_transform(), section_rows=[2])
    rows = [r for rows in ld._block_row_lists() for r in rows]
    assert rows == [2]
    seen = {int(b.section_row.min()) for b in ld}
    assert seen == {2}


def test_restricted_loader_yields_the_same_batches_as_unrestricted(tmp_path):
    """
    Restriction must change WHICH sections are visited, nothing else.

    `num_neighbors=[-1]` (the full neighbourhood, which is what val/test use)
    rather than a sampled one: neighbour sampling draws from the global torch
    RNG, so an unrestricted loader that has already built section 0 arrives at
    section 1 with a different RNG state and samples a different — equally
    valid — neighbourhood. That difference is not what this test is about.
    """
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    kw = {"K": 1, "transform": _transform(), "num_neighbors": [-1]}
    full = [b for b in _loader(blob, **kw) if int(b.section_row.min()) == 1]
    sub = list(_loader(blob, section_rows=[1], **kw))
    assert len(full) == len(sub) and len(sub) > 0
    for a, b in zip(full, sub, strict=True):
        assert int(a.batch_size) == int(b.batch_size)
        assert torch.equal(a.x, b.x)
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.n_id, b.n_id)


def test_len_counts_only_the_restricted_rows(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    sub = _loader(blob, K=1, transform=_transform(), section_rows=[0])
    assert len(sub) == len(list(sub))


def test_out_of_range_and_empty_section_rows_raise(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    with pytest.raises(IndexError):
        _loader(blob, section_rows=[99])
    with pytest.raises(ValueError, match="empty"):
        _loader(blob, section_rows=[])


def test_unrestricted_behaviour_is_unchanged(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    ld = _loader(blob, K=2, transform=_transform())
    assert ld.section_rows is None
    assert [r for rows in ld._block_row_lists() for r in rows] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# whole-section splits through the DataModule
# --------------------------------------------------------------------------- #

def _dm(blob, **kw):
    return OnDiskStreamingDataModule(
        dataset=blob, edge_index_name=EDGE, sections_per_block=1,
        batch_size=4, num_neighbors=[2], section_transform=_transform(),
        prefetch=False, **kw,
    )


def test_split_sections_scopes_val_and_leaves_the_rest_to_train(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    dm = _dm(blob, split_sections={"val": ["section_2.h5ad"]})
    assert dm.val_dataloader().section_rows == [2]
    assert dm.train_dataloader().section_rows == [0, 1]


def test_named_split_overrides_the_cell_mask(tmp_path):
    """Whole-section and in-section splits are two answers to the same
    question; applying both would silently intersect them."""
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    dm = _dm(blob, split_sections={"val": ["section_2.h5ad"]})
    assert dm.val_dataloader().input_mask_attr is None


def test_no_split_sections_keeps_the_mask_path(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    dm = _dm(blob)
    ld = dm.val_dataloader()
    assert ld.section_rows is None and ld.input_mask_attr == "val_mask"


# --------------------------------------------------------------------------- #
# the zero-shot batch map — every held-out section must be UNSEEN, and every
# embedding row must be trained
# --------------------------------------------------------------------------- #

def test_map_restricted_to_rows(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True,
                  batch_key="dataset_batch")
    assert len(blob.batch_label_to_dense()) == 3
    assert blob.batch_label_to_dense(rows=[0, 1]) == {
        "xp_batch0": 0, "xp_batch1": 1}


def test_train_only_map_leaves_no_untrained_rows(tmp_path):
    """
    Derived over the whole blob the embedding would carry one untrained row per
    held-out section; derived over train rows every row trains.
    """
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True,
                  batch_key="dataset_batch")
    dm = _dm(blob, split_sections={"val": ["section_2.h5ad"]})
    assert len(dm.batch_label_to_dense) == 2          # not 3
    assert "xp_batch2" not in dm.batch_label_to_dense


def test_heldout_sections_are_marked_unseen(tmp_path):
    """The held-out section's label is absent from the train map, so its cells
    take the mean-embedding path instead of an arbitrary reference batch."""
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True,
                  batch_key="dataset_batch")
    dm = _dm(blob, split_sections={"val": ["section_2.h5ad"]})
    val = dm.val_dataloader()
    seen_flags = {bool(b.adata_batch_ids_unseen_mask.all()) for b in val}
    assert seen_flags == {True}
    train = dm.train_dataloader()
    assert not any(b.adata_batch_ids_unseen_mask.any() for b in train)


def test_train_ids_stay_within_the_embedding(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True,
                  batch_key="dataset_batch")
    dm = _dm(blob, split_sections={"val": ["section_2.h5ad"]})
    n = len(dm.batch_label_to_dense)
    for b in dm.train_dataloader():
        assert int(b.adata_batch_ids.max()) < n


def test_explicit_map_still_wins(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True,
                  batch_key="dataset_batch")
    forced = {"xp_batch0": 0}
    dm = _dm(blob, split_sections={"val": ["section_2.h5ad"]},
             batch_label_to_dense=forced)
    assert dm.batch_label_to_dense == forced


def test_whole_blob_map_when_no_named_splits(tmp_path):
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True,
                  batch_key="dataset_batch")
    assert len(_dm(blob).batch_label_to_dense) == 3


# --------------------------------------------------------------------------- #
# the reconstruction-vs-graph balance diagnostic
# --------------------------------------------------------------------------- #

class _Recorder:
    """Minimal stand-in for a LightningModule: captures `self.log` calls."""

    def __init__(self):
        self.logged = {}

    def log(self, name, value, **kw):
        self.logged[name] = float(value)


def _balance(nb, adj, loss_data, mode="train"):
    from vqniche.models.base_model import BaseModel
    r = _Recorder()
    BaseModel._log_recon_graph_balance(
        r, torch.tensor(float(nb)), torch.tensor(float(adj)),
        loss_data, mode, 8,
    )
    return r.logged


def test_balance_ratio_is_reconstruction_over_graph():
    got = _balance(90.0, 30.0, {})
    assert got["train_recon_over_graph"] == pytest.approx(3.0)


def test_panel_width_comes_from_the_mask_when_cross_panel():
    """A block mixes panels, so per-cell measured genes is the only
    meaningful width."""
    mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
    got = _balance(10.0, 5.0, {"gene_mask": mask})
    assert got["train_panel_width_mean"] == pytest.approx(2.5)   # (3+2)/2


def test_panel_width_falls_back_to_input_width():
    got = _balance(10.0, 5.0, {"x": torch.zeros(6, 169)})
    assert got["train_panel_width_mean"] == pytest.approx(169.0)


def test_no_graph_term_logs_nothing():
    """Variants without an adjacency loss must not divide by zero."""
    assert _balance(10.0, 0.0, {"x": torch.zeros(6, 169)}) == {}


def test_mode_prefixes_the_metric():
    assert "val_recon_over_graph" in _balance(4.0, 2.0, {}, mode="val")


def test_the_swing_the_diagnostic_exists_to_measure():
    """
    Same graph term, panel widths 169 vs 5049: the NB term scales with measured
    genes, so the realised balance moves ~30x. This is the number that decides
    whether `wt_adj_reconstr=1000` still means what it meant at ~5,000 genes.
    """
    narrow = _balance(169.0, 100.0, {"x": torch.zeros(4, 169)})
    wide = _balance(5049.0, 100.0, {"x": torch.zeros(4, 5049)})
    ratio = wide["train_recon_over_graph"] / narrow["train_recon_over_graph"]
    assert ratio == pytest.approx(5049 / 169, rel=1e-3)


# --------------------------------------------------------------------------- #
# the silent-fallback leak the dry run caught
# --------------------------------------------------------------------------- #

def test_unnamed_split_raises_instead_of_leaking(tmp_path):
    """
    With whole-section splits configured but a split unnamed, the loader used
    to fall back to an in-section cell mask over EVERY section -- drawing that
    split's cells from the training sections. The run looked healthy (`val_*`
    appeared and improved), so this must raise rather than warn.
    """
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    dm = _dm(blob, split_sections={"test": ["section_2.h5ad"]})
    with pytest.raises(ValueError, match="names no sections for 'val'"):
        dm.val_dataloader()
    # test IS named, so it resolves; train takes the remainder.
    assert dm.test_dataloader().section_rows == [2]
    assert dm.train_dataloader().section_rows == [0, 1]


def test_predict_still_spans_everything(tmp_path):
    """predict has no mask, so it is not a leak risk and must not raise."""
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    dm = _dm(blob, split_sections={"test": ["section_2.h5ad"]})
    ld = dm.predict_dataloader()
    assert ld.input_mask_attr is None and ld.section_rows is None


# --------------------------------------------------------------------------- #
# partially-present labels — a blob that builds and then fails at training
# --------------------------------------------------------------------------- #

def _write_mixed_labels(root, name="mix"):
    """Section 0 carries `cell_type`, section 1 does not — the corpus case,
    where 143 of 636 sections have it."""
    import anndata as ad

    rng = np.random.default_rng(0)
    silver = root / "silver" / name
    silver.mkdir(parents=True, exist_ok=True)
    for i, has_label in enumerate([True, False]):
        n = 12 + i
        a = ad.AnnData(np.tile(np.arange(1, 4, dtype="float32"), (n, 1)))
        a.var.index = ["a", "b", "c"]
        a.obs["cell_id"] = [f"b{i}_c{j}" for j in range(n)]
        if has_label:
            a.obs["cell_type"] = rng.choice(["A", "B"], size=n)
        a.obsm["spatial"] = rng.random((n, 2)) * 100
        a.uns.update(batch=f"batch{i}", dataset_id=name, tissue="t", species="s")
        a.write_h5ad(silver / f"section_{i}.h5ad")
    return name


def test_partially_present_label_is_refused_at_build_time(tmp_path):
    """
    Building would succeed and TRAINING would fail — after ~48 h for the
    corpus. PyG's collate takes its key set from the first Data in a block, so
    a block mixing labelled and unlabelled sections raises KeyError, and one
    where the unlabelled section comes first silently trains without labels.
    """
    name = _write_mixed_labels(tmp_path)
    with pytest.raises(ValueError, match="present on 1 section"):
        _build(tmp_path, name, cross_panel=True)


def test_the_refusal_names_both_ways_out(tmp_path):
    name = _write_mixed_labels(tmp_path)
    with pytest.raises(ValueError) as e:
        _build(tmp_path, name, cross_panel=True)
    msg = str(e.value)
    assert "label_names" in msg and "exclude_sections" in msg
    # and states why dropping the label costs nothing
    assert "sidecar" in msg and "NMI/ARI" in msg


def test_no_labels_at_all_is_fine(tmp_path):
    """`label_names=[]` is the corpus setting: consistent across sections."""
    name = _write_mixed_labels(tmp_path)
    blob = OnDiskDatasetBlob(
        name=name, feature_names=["cell_gene_counts"], label_names=[],
        graph_kwargs=GRAPH_KWARGS, data_directory_path=tmp_path,
        pre_filter=None, overwrite=True,
        software_paths={"deepwalk": "", "gosh": ""}, cross_panel=True,
    )
    assert len(blob) == 2
    for r in range(2):
        assert not any(k.startswith("y_") for k in blob.get(r).keys())


def test_label_on_every_section_still_works(tmp_path):
    """The guard must not fire when the label is uniformly present."""
    blob = _build(_write(tmp_path, PANELS) and tmp_path, "xp", cross_panel=True)
    assert len(blob) == 3
    assert "y_cell_types" in blob.get(0).keys()
