"""
Tests for the transform-scope guard on the IN-MEMORY path.

Both backends apply their composed transform per section — the in-memory path via PyG's
`Dataset.__getitem__` (`torch_geometric/data/dataset.py:291`), reached through
`initialize_databatch`'s `[dataset_blob[idx] for idx in adata_batch_idx]`, and the streaming
loader via `section_transform`. So `SubsetHVG` selects a *different* gene set per section on
either path, and because every resulting `x` shares a width, collation succeeds while column j
stops meaning the same gene.

The streaming loader has refused this since `1630f27`. These tests cover the in-memory half, so
the older and more-used path is no longer the less protected one, and pin the boundary: the guard
must fire only when more than one section will actually be loaded.
"""

import anndata as ad
import numpy as np
import pytest
import torch_geometric.transforms as T

from vqniche.dataset.in_memory_dataset_blob import InMemoryDatasetBlob
from vqniche.dataset.transform_scope import _reject_global_scope_transforms
from vqniche.dataset.transforms import SubsetHVG
from vqniche.initializers.initialize import initialize_dataset_blob

SECTION_SIZES = (31, 24, 19)
N_GENES = 16
GRAPH_KWARGS = {
    "coord_type": "generic",
    "spatial_key": "spatial",
    "n_neighs_list": [4],
    "radius_list": None,
    "include_self_loop": True,
    "k": {},
}


def _write_silver(root, name="tsds", sizes=SECTION_SIZES, seed=0):
    """Tiny synthetic silver corpus; mirrors the helper in the other test modules."""
    rng = np.random.default_rng(seed)
    silver = root / "silver" / name
    silver.mkdir(parents=True)
    for i, n in enumerate(sizes):
        adata = ad.AnnData(rng.poisson(0.8, size=(n, N_GENES)).astype("float32"))
        adata.var.index = [f"gene{j}" for j in range(N_GENES)]
        adata.obs["cell_type"] = rng.choice(["A", "B"], size=n)
        adata.obs["cell_id"] = [f"b{i}_c{j}" for j in range(n)]
        adata.obsm["spatial"] = rng.random((n, 2)) * 50
        adata.uns["batch"] = f"batch{i}"
        adata.uns["dataset_id"] = name
        adata.uns["tissue"] = "synthetic"
        adata.uns["species"] = "synthetic"
        adata.write_h5ad(silver / f"section_{i}.h5ad")
    return name


def _config(root, name, *, hvg, adata_batch_idx):
    """
    The minimum `initialize_dataset_blob` reads. Mirrors what `train()` materialises: it
    resolves `apply_hvg` into `gene_count_transform_names` before the initializers see it
    (run_squint.py, "Resolve HVG transform based on apply_hvg flag").
    """
    return {
        "dataset": {
            "dataset_name": name,
            "root_data_dir": str(root),
            "adata_batch_idx": adata_batch_idx,
            "gene_count_transform_names": ["SubsetHVG"] if hvg else [],
            "gene_count_transform_params": {"n_genes": 8} if hvg else {},
            # SetExperimentDataKeys namespace ("X"), as the training configs use
            # (run_squint.py:968) -- not the blob-build namespace
            "feature_names": ["X"],
            "label_name": "cell_types",
            "graph_params": {"spatial_key": "spatial", "delaunay": False,
                             "n_neighs": 4, "radius": None},
            "train_transform_names": [],
            "train_transform_params": {},
        },
        "model": {"encoder_params": {}},
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """
    Silver files PLUS a built in-memory blob.

    `initialize_dataset_blob` constructs `InMemoryDatasetBlob` with default graph_kwargs and
    expects to LOAD an already-built `dataset_blob.pt` (the build is a separate step, via
    `--build-blob` or `analysis/create_in_memory_dataset_blob.py`). So the blob has to exist
    first, otherwise it tries to process and fails on the missing graph config.
    """
    root = tmp_path_factory.mktemp("ts_corpus")
    name = _write_silver(root)
    InMemoryDatasetBlob(
        name=name,
        feature_names=["cell_gene_counts"],
        label_names=["cell_types=cell_type"],
        graph_kwargs=GRAPH_KWARGS,
        data_directory_path=root,
        pre_transform=None,
        pre_filter=None,
        overwrite=True,
        software_paths={"deepwalk": "", "gosh": ""},
    )
    return root, name


# --------------------------------------------------------------------------- #
# the guard, through the real entry point
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "adata_batch_idx",
    [-1, [0, 1], [0, 1, 2]],
    ids=["all-sections", "two-explicit", "three-explicit"],
)
def test_hvg_is_refused_when_multiple_sections_load(corpus, adata_batch_idx):
    """
    The regression test. `-1` means every section, so it must be resolved against the blob
    length rather than assumed single.
    """
    root, name = corpus
    cfg = _config(root, name, hvg=True, adata_batch_idx=adata_batch_idx)
    with pytest.raises(ValueError, match="single decision for the whole corpus"):
        initialize_dataset_blob(cfg)


def test_error_names_both_backends(corpus):
    """
    The message must not imply this is streaming-only — that misreading is what left the
    in-memory path unguarded in the first place.
    """
    root, name = corpus
    cfg = _config(root, name, hvg=True, adata_batch_idx=-1)
    with pytest.raises(ValueError) as exc:
        initialize_dataset_blob(cfg)
    msg = str(exc.value)
    assert "BOTH backends" in msg
    assert "Dataset.__getitem__" in msg      # names the actual scoping mechanism
    assert "SubsetHVG" in msg                # names the offender
    assert "apply_hvg=False" in msg          # says what to do instead


def test_single_section_is_allowed(corpus):
    """
    With one section, per-section selection is trivially self-consistent, so the guard must
    NOT fire — counting exactly is the point.
    """
    root, name = corpus
    blob = initialize_dataset_blob(_config(root, name, hvg=True, adata_batch_idx=[0]))
    assert blob is not None
    assert len(blob) == len(SECTION_SIZES)   # blob holds all sections; only one is loaded


def test_no_hvg_is_allowed_on_many_sections(corpus):
    """The default path (apply_hvg=False) must be untouched."""
    root, name = corpus
    blob = initialize_dataset_blob(_config(root, name, hvg=False, adata_batch_idx=-1))
    assert blob is not None
    assert len(blob) == len(SECTION_SIZES)


# --------------------------------------------------------------------------- #
# the shared helper itself
# --------------------------------------------------------------------------- #

def test_guard_finds_nested_offenders():
    with pytest.raises(ValueError, match="SubsetHVG"):
        _reject_global_scope_transforms(
            T.Compose([T.Compose([SubsetHVG(n_genes=4)]), T.NormalizeFeatures()])
        )


def test_guard_allows_per_section_transforms():
    _reject_global_scope_transforms(None)
    _reject_global_scope_transforms(T.RandomNodeSplit(num_val=0.1, num_test=0.1))
    _reject_global_scope_transforms(
        T.Compose([T.RandomNodeSplit(num_val=0.1, num_test=0.1), T.NormalizeFeatures()])
    )


def test_where_argument_appears_in_the_message():
    """Each caller names its own slot, so the error points at the right knob."""
    with pytest.raises(ValueError, match="my-custom-slot"):
        _reject_global_scope_transforms(SubsetHVG(n_genes=4), where="my-custom-slot")


def test_streaming_reexport_is_the_same_object():
    """`on_disk_dataset` re-exports the guard, so existing importers keep working."""
    from vqniche.dataset import on_disk_dataset

    assert on_disk_dataset._reject_global_scope_transforms is _reject_global_scope_transforms
