"""VQNiche / SQUINT — tokenization for tissue sections."""

# Deliberately the ONLY thing this module does. `b64a999` emptied this file to make the
# relative submodule imports lazy (importing `vqniche` must not drag in torch, scanpy,
# squidpy, ...), so nothing here may import from the package itself. `importlib.metadata`
# is stdlib and cheap, so exposing the version does not reintroduce that cost.
#
# Guarded because the package is NOT always installed: every `_*_job.sh` and the
# `examples/submit_*.sh` wrappers run with `PYTHONPATH=<repo>/src` against an interpreter
# that may have no `vqniche` distribution metadata. An unguarded lookup would raise
# PackageNotFoundError at import time and take down every one of those jobs.
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("vqniche")
except PackageNotFoundError:  # imported from source without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
