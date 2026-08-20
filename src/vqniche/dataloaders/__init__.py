from . import (
    in_memory_datamodule,
)

# Streaming (on-disk) DataModule lives with the on-disk dataset; re-export here
# so both datamodules are reachable from vqniche.dataloaders.
from ..dataset.on_disk_dataset import OnDiskStreamingDataModule
