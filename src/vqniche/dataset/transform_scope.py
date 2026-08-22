"""
Which PyG transforms may be applied per section, and which must decide once for the corpus.

BOTH dataset backends apply their composed transform **per section**, and this is easy to
misread:

  - in-memory: `initialize_dataset_blob` hands the composed transform to the dataset as
    `transform=`, and PyG applies it inside `Dataset.__getitem__`
    (torch_geometric/data/dataset.py:291). `initialize_databatch` then builds its list with
    `[dataset_blob[idx] for idx in adata_batch_idx]`, so the transform has already run on each
    section BEFORE collation.
  - streaming: `KSectionBlockLoader` applies `section_transform` to each section before
    concatenating a block.

Identical scoping. Most transforms are fine with it — `RandomNodeSplit` / `SpatialBatchSplit`
produce within-section splits by design, and `SetExperimentDataKeys` just renames keys. But a
transform that has to make ONE decision for the whole corpus is silently wrong per section.

`SubsetHVG` is the case that matters (dataset/transforms.py:516-530): it picks the top-n highly
variable genes from whichever single `Data` it is handed, so per section it selects a DIFFERENT
gene set per section. Every resulting `x` has width n_genes, so collation succeeds without
complaint even though column j means a different gene in different sections — and since the model
has one weight per input column shared across all sections, that weight is then trained on two
unrelated genes at once. Nothing raises; the run completes and the features are meaningless.

It has never fired only because `apply_hvg` defaults to False (the panels are already curated),
not because either path protected it.

Fixing it properly means choosing the gene set once, over the whole corpus, and applying the SAME
indices to every section. Until that exists, refuse the transform rather than silently corrupting
features.
"""

from __future__ import annotations

_GLOBAL_SCOPE_TRANSFORMS = {
    "SubsetHVG": (
        "picks highly variable genes from the Data it is given, so per-section "
        "application selects a different gene set per section and column j "
        "stops meaning the same gene across the corpus"
    ),
}


def _iter_transforms(transform):
    """Yield a transform and, for a Compose, each of its members (recursively)."""
    if transform is None:
        return
    inner = getattr(transform, "transforms", None)
    if inner is not None:
        for t in inner:
            yield from _iter_transforms(t)
    else:
        yield transform


def _reject_global_scope_transforms(transform, where: str = "section_transform") -> None:
    """
    Raise if `transform` contains a transform that must decide for the whole corpus.

    Converts a silent feature-scrambling bug into an immediate, explanatory error.

    Parameters
    ----------
    transform
        A transform, a `Compose`, or None. Nested `Compose` is searched.
    where
        How to name the offending slot in the error, so each caller can describe its own
        context (the streaming `section_transform`, or the in-memory composed dataset
        transform).
    """
    offenders = [
        (type(t).__name__, _GLOBAL_SCOPE_TRANSFORMS[type(t).__name__])
        for t in _iter_transforms(transform)
        if type(t).__name__ in _GLOBAL_SCOPE_TRANSFORMS
    ]
    if not offenders:
        return
    lines = "\n".join(f"  - {name}: {why}" for name, why in offenders)
    raise ValueError(
        f"{where} contains transform(s) that must make a single decision for the "
        f"whole corpus and cannot be applied per section:\n"
        f"{lines}\n"
        "This affects BOTH backends: the in-memory path applies its composed transform in "
        "PyG's Dataset.__getitem__ (per section, before initialize_databatch collates), and "
        "the streaming loader applies section_transform per section before building a block. "
        "Neither would raise at runtime — the shapes stay consistent while the meaning of each "
        "feature column diverges between sections — so it is rejected here instead.\n"
        "For highly variable genes: either train on the full panel (apply_hvg=False, the "
        "default), or select the gene set once over the whole corpus and apply the same indices "
        "to every section. Split transforms (RandomNodeSplit, SpatialBatchSplit) and "
        "SetExperimentDataKeys are per-section by design and remain fine."
    )
