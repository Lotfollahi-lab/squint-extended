"""
Tests for the filesystem storage container (`vqniche.dataset.file_database`).

The container exists because SQLite caps a single value at its compile-time `MAX_LENGTH`
(1e9 bytes on this stack) while real tissue sections reach ~7 GB as one pickled blob, so the
corpus could not be stored at all. These tests pin:

  - the two containers are interchangeable (same `get(idx)`, same manifest-derived metadata,
    same streaming behaviour) — so switching costs nothing;
  - `FileDatabase` accepts a value that SQLite rejects — the direct regression test for the
    blocker;
  - a blob written before the `container` field existed still opens, as sqlite.

The oversized-value test needs several GB of RAM, so it is opt-in via `SQUINT_SLOW_TESTS=1`
(set by the LSF job) and skipped by default. Everything else runs in seconds on tiny
synthetic sections.
"""

import json
import os

import anndata as ad
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from vqniche.dataset.file_database import FileDatabase
from vqniche.dataset.on_disk_dataset import KSectionBlockLoader, OnDiskDatasetBlob

SECTION_SIZES = (37, 23, 41)
N_GENES = 12
BATCH_SIZE = 8
GRAPH_KWARGS = {
    "coord_type": "generic",
    "spatial_key": "spatial",
    "n_neighs_list": [4],
    "radius_list": None,
    "include_self_loop": True,
    "k": {},          # no spectral embeddings — nothing consumes them
}

SLOW = pytest.mark.skipif(
    os.environ.get("SQUINT_SLOW_TESTS") != "1",
    reason="needs several GB of RAM; set SQUINT_SLOW_TESTS=1 (the LSF job does)",
)


def _write_silver(root, name="ccds", sizes=SECTION_SIZES, seed=0):
    """
    Write a tiny synthetic silver corpus.

    Mirrors the helper in `test_on_disk_streaming.py`; kept local so each test module stays
    self-contained and independently runnable.
    """
    rng = np.random.default_rng(seed)
    silver = root / "silver" / name
    silver.mkdir(parents=True)
    for i, n in enumerate(sizes):
        adata = ad.AnnData(rng.poisson(0.6, size=(n, N_GENES)).astype("float32"))
        adata.var.index = [f"gene{j}" for j in range(N_GENES)]
        adata.obs["cell_type"] = rng.choice(["A", "B", "C"], size=n)
        adata.obs["cell_id"] = [f"b{i}_c{j}" for j in range(n)]
        adata.obsm["spatial"] = rng.random((n, 2)) * 100
        adata.uns["batch"] = f"batch{i}"
        adata.uns["dataset_id"] = name
        adata.uns["tissue"] = "synthetic"
        adata.uns["species"] = "synthetic"
        adata.write_h5ad(silver / f"section_{i}.h5ad")
    return name


def _build(root, name, backend, overwrite=True):
    return OnDiskDatasetBlob(
        name=name,
        feature_names=["cell_gene_counts"],
        label_names=["cell_types=cell_type"],
        graph_kwargs=GRAPH_KWARGS,
        data_directory_path=root,
        pre_filter=None,
        overwrite=overwrite,
        software_paths={"deepwalk": "", "gosh": ""},
        backend=backend,
    )


# --------------------------------------------------------------------------- #
# FileDatabase in isolation
# --------------------------------------------------------------------------- #

def test_empty_store_has_zero_length(tmp_path):
    """`OnDiskDataset.db` calls len() straight after construction, before any insert."""
    db = FileDatabase(path=str(tmp_path / "sections"))
    assert len(db) == 0
    assert db.row_indices() == []


def test_roundtrip_preserves_list_attrs(tmp_path):
    """Python-list node attrs must survive, as they do under the pickle schema."""
    db = FileDatabase(path=str(tmp_path / "sections"))
    db.insert(0, Data(x=torch.rand(4, 3), cell_id=["a", "b", "c", "d"]))
    got = db.get(0)
    assert got.cell_id == ["a", "b", "c", "d"]
    assert got.x.shape == (4, 3)


def test_missing_row_raises_indexerror(tmp_path):
    db = FileDatabase(path=str(tmp_path / "sections"))
    db.insert(0, Data(x=torch.rand(2, 2)))
    with pytest.raises(IndexError, match="No row 5"):
        db.get(5)


def test_partial_writes_are_not_counted(tmp_path):
    """
    `insert` writes to `.partial` then renames, so an interrupted build leaves either a
    complete row or none. A stray `.partial` must not inflate the length.
    """
    d = tmp_path / "sections"
    db = FileDatabase(path=str(d))
    db.insert(0, Data(x=torch.rand(2, 2)))
    (d / "section_00001.pt.partial").write_bytes(b"truncated")
    assert len(db) == 1
    assert db.row_indices() == [0]


def test_length_survives_a_fresh_handle(tmp_path):
    """Length is derived from the directory, so it is correct across process restarts."""
    d = str(tmp_path / "sections")
    first = FileDatabase(path=d)
    for i in range(3):
        first.insert(i, Data(x=torch.rand(2, 2)))
    assert len(FileDatabase(path=d)) == 3


# --------------------------------------------------------------------------- #
# The blocker: values SQLite cannot store
# --------------------------------------------------------------------------- #

@SLOW
def test_file_container_accepts_a_value_sqlite_rejects(tmp_path):
    """
    The direct regression test for the reason this container exists.

    ~1.05 GB is just over SQLite's 1e9-byte `MAX_LENGTH`; real sections are ~7x over.
    """
    from torch_geometric.data.database import SQLiteDatabase

    big = Data(x=torch.zeros(int(1.05 * 1024**3) // 4, dtype=torch.float32))

    fdb = FileDatabase(path=str(tmp_path / "sections"))
    fdb.insert(0, big)
    assert fdb.get(0).x.numel() == big.x.numel()

    sdb = SQLiteDatabase(path=str(tmp_path / "s.db"), name="data", schema=object)
    with pytest.raises(Exception) as exc:      # sqlite3.InterfaceError
        sdb.insert(0, big)
    assert "InterfaceError" in type(exc.value).__name__ or "bind" in str(exc.value).lower()
    sdb.close()


# --------------------------------------------------------------------------- #
# The two containers must be interchangeable
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def both(tmp_path_factory):
    """Build the same corpus under each container."""
    out = {}
    for backend in ("sqlite", "file"):
        root = tmp_path_factory.mktemp(f"corpus_{backend}")
        name = _write_silver(root)
        out[backend] = (root, name, _build(root, name, backend))
    return out


@pytest.mark.parametrize("backend", ["sqlite", "file"])
def test_manifest_records_the_container(both, backend):
    _, _, blob = both[backend]
    manifest = json.loads((blob._meta_dir / "manifest.json").read_text())
    assert manifest["container"] == backend


def test_store_layout_matches_the_backend(both):
    """sqlite writes one file; the file container writes a directory of per-section files."""
    _, _, sq = both["sqlite"]
    _, _, fl = both["file"]
    assert sq.processed_file_names == ["sqlite.db", "manifest.json"]
    assert fl.processed_file_names == ["sections", "manifest.json"]
    assert os.path.isfile(sq.processed_paths[0])
    assert os.path.isdir(fl.processed_paths[0])
    assert sorted(os.listdir(fl.processed_paths[0])) == [
        f"section_{i:05d}.pt" for i in range(len(SECTION_SIZES))
    ]


def test_sections_are_identical_across_containers(both):
    """Same rows, same tensors — switching container must not change the data."""
    _, _, sq = both["sqlite"]
    _, _, fl = both["file"]
    assert len(sq) == len(fl) == len(SECTION_SIZES)
    for i in range(len(sq)):
        a, b = sq.get(i), fl.get(i)
        assert sorted(a.keys()) == sorted(b.keys())
        for k in a.keys():
            av, bv = a[k], b[k]
            if torch.is_tensor(av):
                assert torch.equal(av, bv), f"row {i} attr {k} differs"
            else:
                assert av == bv, f"row {i} attr {k} differs"


def test_manifest_metadata_is_identical(both):
    _, _, sq = both["sqlite"]
    _, _, fl = both["file"]
    assert sq.get_node_counts() == fl.get_node_counts() == list(SECTION_SIZES)
    assert sq.get_batch_labels() == fl.get_batch_labels()
    assert sq.batch_label_to_dense() == fl.batch_label_to_dense()


@pytest.mark.parametrize("K", [1, 2, 3])
def test_streaming_is_identical_across_containers(both, K):
    """
    The loader must behave the same on either container, including with prefetching on —
    which is where the sqlite path needs a per-thread handle and the file path does not.
    """
    def fingerprint(blob):
        ld = KSectionBlockLoader(
            dataset=blob, edge_index_name="spatial_n_neighs_4",
            sections_per_block=K, batch_size=BATCH_SIZE, num_neighbors=[2],
            shuffle=False, prefetch=True,
            batch_label_to_dense=blob.batch_label_to_dense(),
        )
        # input_id.numel() is the ground-truth seed count, not batch_size
        seeds = [(int(mb.input_id.numel()),
                  mb.adata_batch_ids[:int(mb.input_id.numel())].tolist())
                 for mb in ld]
        return len(ld), seeds

    _, _, sq = both["sqlite"]
    _, _, fl = both["file"]
    assert fingerprint(sq) == fingerprint(fl)


# --------------------------------------------------------------------------- #
# Back-compat
# --------------------------------------------------------------------------- #

def test_blob_without_container_field_opens_as_sqlite(tmp_path):
    """Blobs built before the container field existed must keep working."""
    root = tmp_path / "old"
    root.mkdir()
    name = _write_silver(root)
    blob = _build(root, name, "sqlite")

    mpath = blob._meta_dir / "manifest.json"
    manifest = json.loads(mpath.read_text())
    del manifest["container"]
    mpath.write_text(json.dumps(manifest))

    reopened = _build(root, name, "sqlite", overwrite=False)
    assert reopened.container == "sqlite"
    assert len(reopened) == len(SECTION_SIZES)
    assert reopened.get_node_counts() == list(SECTION_SIZES)
