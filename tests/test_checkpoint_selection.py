"""
Checkpoint selection is a POLICY, not a lookup.

`save_top_k=-1` leaves one checkpoint per validation pass -- ten on the corpus
budget -- and which one a benchmark should use is a decision. The retained
"best" is best by `val_loss`, a weighted sum whose weights were tuned for
optimisation dynamics, 55% of which is a stochastic adjacency term, and whose
components move in opposite directions: on the first corpus run it selected step
40,000 of 200,000 while reconstruction was still improving. So `best` and "best
for reconstruction" were different checkpoints, and the five benchmark regimes
may each prefer a different one.
"""
import importlib.util
import sys

import pytest


def _rs():
    """Load the driver as a module without running its CLI."""
    sys.argv = ["run_squint.py"]
    spec = importlib.util.spec_from_file_location("rs", "examples/run_squint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rs():
    return _rs()


def _run_with(tmp_path, names, last=True):
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")
    if last:
        (d / "last.ckpt").write_bytes(b"x")
    return tmp_path


# What training now writes: epoch, step, then the monitored metric.
TEN = [f"epoch=0-step={s}-val_loss={4000 - s // 100:.3f}.ckpt"
       for s in range(20_000, 220_000, 20_000)]


def test_best_defers_to_the_existing_finder(rs, tmp_path):
    """`best` returns None so `find_best_checkpoint` runs, unchanged."""
    run = _run_with(tmp_path, TEN)
    assert rs._resolve_checkpoint(run, "best") is None
    assert rs._resolve_checkpoint(run, None) is None


def test_last_selects_last_ckpt(rs, tmp_path):
    run = _run_with(tmp_path, TEN)
    assert rs._resolve_checkpoint(run, "last").endswith("last.ckpt")


def test_last_raises_when_save_last_was_off(rs, tmp_path):
    run = _run_with(tmp_path, TEN, last=False)
    with pytest.raises(FileNotFoundError, match="no last.ckpt"):
        rs._resolve_checkpoint(run, "last")


def test_step_selects_that_step(rs, tmp_path):
    run = _run_with(tmp_path, TEN)
    got = rs._resolve_checkpoint(run, "step:40000")
    assert "step=40000-" in got
    # and it is NOT the one `best` would pick, which is the point
    assert "step=200000" not in got


def test_step_not_present_lists_what_is(rs, tmp_path):
    run = _run_with(tmp_path, TEN)
    with pytest.raises(FileNotFoundError) as e:
        rs._resolve_checkpoint(run, "step:12345")
    msg = str(e.value)
    assert "Steps present" in msg and "40000" in msg


def test_step_on_pre_step_stamped_checkpoints_says_so(rs, tmp_path):
    """Runs before step-stamped filenames cannot be addressed by step, and the
    error should say that rather than just 'not found'."""
    run = _run_with(tmp_path, ["epoch=0-val_loss=3220.712.ckpt"])
    with pytest.raises(FileNotFoundError, match="predate step-stamped"):
        rs._resolve_checkpoint(run, "step:40000")


def test_an_explicit_path_passes_through(rs, tmp_path):
    run = _run_with(tmp_path, TEN)
    p = str(run / "checkpoints" / TEN[0])
    assert rs._resolve_checkpoint(run, p) == p


def test_a_bogus_policy_raises(rs, tmp_path):
    run = _run_with(tmp_path, TEN)
    with pytest.raises(FileNotFoundError, match="neither a policy"):
        rs._resolve_checkpoint(run, "penultimate")


def test_missing_checkpoints_dir_raises(rs, tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoints/"):
        rs._resolve_checkpoint(tmp_path, "last")


def test_step_filename_is_parseable_by_find_best_checkpoint(rs, tmp_path):
    """The added `step` field must not break metric parsing, which splits on
    f'{metric_name}=' and takes what follows."""
    from vqniche.utils.parse_test_configs import find_best_checkpoint
    run = _run_with(tmp_path, TEN, last=False)
    got = find_best_checkpoint(str(run), mode="min", metric_name="val_loss")
    # returns a Path, not a str
    assert "step=200000" in str(got)
