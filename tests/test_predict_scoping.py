"""
Predict must touch only the sections it was asked to predict.

`--predict-sections` restricts `config['dataset']['adata_batch_idx']` to a rung's
blob rows. Two loops in `predict()` then scanned the WHOLE blob regardless, and
the first of them raised:

    RuntimeError: Could not match blob adata_batch_ids [0..20, 32..635] to any
    silver file via cell_id intersection.

which reads as a corrupt blob and is not one. Ids 21-31 -- chr78's 11 sections,
the rung -- matched perfectly; the other 625 were compared against a `--silver-dir`
that only ever held the rung's 11 files. The scan was the bug, not the data.

The cost is the other half. `dataset_blob[pos]` fully deserialises a section
(counts, edges, 128-dim Laplacian eigenvectors per graph), so an unscoped scan
pays ~944 GB of I/O on the corpus to use 11 sections' worth of it.

These tests read the driver's source: `predict()` needs a trained checkpoint and
a built blob, so it cannot be called here, but the loop bounds are a static
property and a regression would be silent otherwise.
"""

import ast
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "examples" / "run_squint.py"


@pytest.fixture(scope="module")
def predict_fn():
    tree = ast.parse(DRIVER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            return node
    pytest.fail("no `predict` function in run_squint.py")


def _loops_assigning(fn, target_name):
    """Every `for` loop whose body assigns to a subscript of `target_name`."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.For):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Assign)
                    and any(isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == target_name
                            for t in sub.targets)):
                out.append(node)
                break
    return out


def _is_full_blob_scan(loop):
    """`for _ in range(len(dataset_blob))`."""
    it = loop.iter
    return (isinstance(it, ast.Call)
            and getattr(it.func, "id", None) == "range"
            and len(it.args) == 1
            and isinstance(it.args[0], ast.Call)
            and getattr(it.args[0].func, "id", None) == "len"
            and getattr(it.args[0].args[0], "id", None) == "dataset_blob")


def test_the_cell_id_matching_scan_is_scoped(predict_fn):
    """The loop that raised. It must not iterate the whole blob."""
    loops = _loops_assigning(predict_fn, "blob_id_to_first_cell")
    assert len(loops) == 1, f"expected one such loop, found {len(loops)}"
    loop = loops[0]
    assert not _is_full_blob_scan(loop), (
        "the cell_id matching scan iterates every blob section again -- it will "
        "raise 'Could not match blob adata_batch_ids' for every section outside "
        "--silver-dir"
    )
    assert getattr(loop.iter, "id", None) == "_scan_positions"


def test_the_scan_falls_back_to_the_whole_blob_when_unrestricted(predict_fn):
    """
    Scoping must not silently narrow an UNrestricted predict. `_scan_positions`
    is `adata_batch_idx` when set and the full range otherwise -- and predict
    sets `adata_batch_idx` to every section when `--predict-sections` is absent,
    so the fallback is belt-and-braces rather than the usual path.
    """
    src = ast.get_source_segment(DRIVER.read_text(), predict_fn)
    assert "_scan_positions = config[\"dataset\"].get(\"adata_batch_idx\")" in src
    assert "_scan_positions = list(range(len(dataset_blob)))" in src


def test_the_label_map_scan_is_skipped_when_the_manifest_supplies_it(predict_fn):
    """
    The train label->dense reconstruction also scanned every section, and on a
    streaming blob its result is then DISCARDED for the manifest map. Ask the
    manifest first; reconstruct only when there is no manifest map.
    """
    loops = [n for n in ast.walk(predict_fn)
             if isinstance(n, ast.For) and _is_full_blob_scan(n)]
    for loop in loops:
        names = {getattr(t, "id", None)
                 for sub in ast.walk(loop) if isinstance(sub, ast.Call)
                 for t in [sub.func]}
        if "_train_labels" in ast.dump(loop):
            # must sit under `if _stream_map is None`
            guarded = any(
                isinstance(anc, ast.If)
                and "_stream_map" in ast.dump(anc.test)
                and loop in list(ast.walk(anc))
                for anc in ast.walk(predict_fn) if isinstance(anc, ast.If)
            )
            assert guarded, (
                "the train-label reconstruction scan runs unconditionally; on a "
                "streaming blob it deserialises every section and is discarded"
            )
            return
    # No such loop at all is also fine -- it means the scan was removed.


def test_no_unguarded_full_blob_scan_survives(predict_fn):
    """
    Backstop: any remaining `range(len(dataset_blob))` loop in predict must be
    guarded, so a future edit cannot reintroduce a corpus-wide scan unnoticed.
    """
    src_text = DRIVER.read_text()
    unguarded = []
    for loop in [n for n in ast.walk(predict_fn)
                 if isinstance(n, ast.For) and _is_full_blob_scan(n)]:
        guarded = any(
            isinstance(anc, ast.If) and loop in list(ast.walk(anc))
            for anc in ast.walk(predict_fn) if isinstance(anc, ast.If)
        )
        if not guarded:
            seg = ast.get_source_segment(src_text, loop) or ""
            unguarded.append(seg.splitlines()[0])
    assert not unguarded, f"unguarded full-blob scans in predict(): {unguarded}"


# --------------------------------------------------------------------------- #
# data_split: a streaming run records its split somewhere else entirely
# --------------------------------------------------------------------------- #

import importlib.util  # noqa: E402
import sys  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    """Import run_squint as a module so its helpers can be called directly."""
    spec = importlib.util.spec_from_file_location("_run_squint_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                    # pragma: no cover
        pytest.skip(f"run_squint.py not importable here: {exc}")
    return mod


# The corpus run's real shape: chr78's 11 sections are TERRA validation, and
# `train_transform_params.test_batches` is None.
RELS = (
    [f"train_ds/sec{i}.h5ad" for i in range(21)]          # rows 0-20  train
    + [f"chr78-11b_1p/CM{i:03d}_annotated.h5ad" for i in range(11)]   # 21-31 val
    + [f"xhs1022-1-55b_1p/adata_batch{i}.h5ad" for i in range(5)]     # 32-36 test
)
SPLIT_SECTIONS = {
    "val": RELS[21:32],
    "test": RELS[32:37],
}


def test_validation_sections_are_not_labelled_train(driver):
    """The bug: every held-out cell tagged `train` because test_batches is empty."""
    pos_to_id = {p: p for p in range(21, 32)}
    got = driver._derive_terra_split_map(RELS, SPLIT_SECTIONS, pos_to_id)
    assert set(got.values()) == {"validation"}
    assert len(got) == 11


def test_validation_is_not_collapsed_into_test(driver):
    """
    Validation selected the checkpoint. Reporting it as `test` would let the
    model-selection set stand in for a generalisation claim -- and on this
    corpus the entire `unseen_assay` rung is validation, so the collapse would
    not be a rounding error.
    """
    pos_to_id = {p: p for p in list(range(21, 32)) + list(range(32, 37))}
    got = driver._derive_terra_split_map(RELS, SPLIT_SECTIONS, pos_to_id)
    assert sum(v == "validation" for v in got.values()) == 11
    assert sum(v == "test" for v in got.values()) == 5
    assert "test" not in {got[i] for i in range(21, 32)}


def test_train_sections_stay_train(driver):
    got = driver._derive_terra_split_map(RELS, SPLIT_SECTIONS, {p: p for p in range(21)})
    assert set(got.values()) == {"train"}


def test_blob_ids_are_not_assumed_equal_to_row_positions(driver):
    """
    Corpus ids were renumbered positionally where source `uns['batch']`
    counters collided (561/636 did), so the map is keyed on the id the cells
    actually carry, not on the row.
    """
    pos_to_id = {21: 900, 22: 901, 32: 902}
    got = driver._derive_terra_split_map(RELS, SPLIT_SECTIONS, pos_to_id)
    assert got == {900: "validation", 901: "validation", 902: "test"}


def test_a_missing_or_malformed_split_section_config_yields_nothing(driver):
    """No split recorded -> no claim about the split, rather than a wrong one."""
    assert driver._derive_terra_split_map(RELS, None, {1: 1}) == {}
    assert driver._derive_terra_split_map(RELS, {}, {1: 1}) == {}
    assert driver._derive_terra_split_map(RELS, SPLIT_SECTIONS, {}) == {}


def test_out_of_range_positions_do_not_raise(driver):
    got = driver._derive_terra_split_map(RELS, SPLIT_SECTIONS, {9999: 7})
    assert got == {7: "train"}


def test_against_the_real_corpus_split_spec(driver):
    """
    The actual `_splitspec.json` and the actual saved training config, if both
    are present: chr78 must come back as validation, not test and not train.
    """
    import json
    spec_path = DRIVER.resolve().parents[1] / "_splitspec.json"
    if not spec_path.exists():
        pytest.skip("no _splitspec.json in this checkout")
    spec = json.loads(spec_path.read_text())
    rung = spec["smoke_rung_sections"]
    rels = sorted(set(spec["splits"]["train"] + spec["splits"]["validation"]
                      + spec["splits"]["test"]))
    ss = {"val": spec["splits"]["validation"], "test": spec["splits"]["test"]}
    pos = {rels.index(r): 1000 + i for i, r in enumerate(rung)}
    got = driver._derive_terra_split_map(rels, ss, pos)
    assert set(got.values()) == {"validation"}, (
        "chr78 -- the cheapest rung -- is TERRA validation; scoring a "
        "generalisation claim on it measures the checkpoint-selection set"
    )


def test_only_terra_test_becomes_data_split_test(driver):
    """
    `compute_inference_metrics.py` folds val into "train" by design. The binary
    `data_split` must therefore carry TERRA test only -- tagging validation as
    test would report the checkpoint-selection set as held out, which is the
    failure this whole derivation exists to prevent.
    """
    src = DRIVER.read_text()
    i = src.index("_split_map = _derive_terra_split_map")
    window = src[i:i + 1200]
    assert '_derive_terra_split_map' in window
    assert 's == "test"' in window, (
        "data_split is being derived from every non-train section; validation "
        "would be reported as held-out test"
    )
    assert 's != "train"' not in window


# --------------------------------------------------------------------------- #
# The snapshot: the predict loader's split_sections is NOT the training split
# --------------------------------------------------------------------------- #

def test_the_training_split_is_snapshotted_before_it_is_overwritten():
    """
    `--predict-sections` sets `split_sections = {"predict": [...]}` for the
    loader, which DESTROYS the saved {val, test} record. The first corpus run
    did exactly that and shipped a predicted_adata with no `terra_split` and
    every held-out cell tagged "train" -- the derivation was correct and had
    nothing left to derive from.

    Ordering is the whole fix, so assert the ordering.
    """
    src = DRIVER.read_text()
    snap = src.index("_saved_split_sections = dict(")
    overwrite = src.index('["split_sections"] = {\n            "predict"')
    use = src.index("_sections_cfg = _saved_split_sections")
    assert snap < overwrite, (
        "the training split is snapshotted after the predict loader overwrites "
        "it -- the snapshot would capture {'predict': [...]}"
    )
    assert overwrite < use


def test_the_split_derivation_does_not_read_the_live_config():
    """
    Reading `config['datamodule']['split_sections']` at derivation time gets the
    loader's `{"predict": [...]}`, not the training split.
    """
    src = DRIVER.read_text()
    i = src.index("_sections_cfg = ")
    line = src[i:src.index("\n", i)]
    assert "_saved_split_sections" in line, line
    assert "config" not in line, line


def test_predict_sections_still_scopes_the_loader():
    """The snapshot must not change what the loader is told to load."""
    src = DRIVER.read_text()
    i = src.index('["split_sections"] = {\n            "predict"')
    window = src[i:i + 200]
    assert '"predict": list(predict_sections)' in window
