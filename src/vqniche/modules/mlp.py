from typing import Optional, List

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP as MLP_Module


def _lin_or_gather(
        lin: torch.nn.Linear,
        x: torch.Tensor,
        gene_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
    """
    Apply `lin` to `x`, gathering its INPUT columns when `gene_ids` is given.

    Cross-panel support: the layer is built at the gene-vocabulary width `V`,
    but a block only carries `W` genes -- the union of its panels -- so the
    weight is indexed down to those columns. Shared by `MLP` and
    `ConditionalMLP` so the two cannot drift apart.

    `gene_ids` arrives as `[1, W]` (see `OnDiskDatasetBlob._stamp_gene_ids` for
    why that shape rather than `[W]`), so it is flattened here; callers should
    not have to remember.
    """
    if gene_ids is None:
        return lin(x)
    ids = gene_ids.reshape(-1)
    if lin.weight.shape[1] != x.shape[-1]:
        # Only meaningful when the layer really is wider than the input.
        return F.linear(x, lin.weight[:, ids], lin.bias)
    # Already the same width (single-panel blob whose vocabulary equals its
    # panel, or a block spanning the whole vocabulary): the gather would be an
    # identity permutation, so skip it and stay bit-identical.
    return lin(x)


def _lin_or_gather_out(
        lin: torch.nn.Linear,
        x: torch.Tensor,
        gene_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
    """
    Apply `lin` to `x`, gathering its OUTPUT rows when `gene_ids` is given.

    The decoder mirror of `_lin_or_gather`: on the decoder side it is the LAST
    layer that is vocabulary-wide, so the weight's rows and the bias are indexed
    instead of the weight's columns.

    Gathering the weight rather than the emitted logits matters for memory:
    slicing `[V, h] -> [W, h]` touches a few thousand floats, whereas letting
    the layer emit V-wide logits and selecting afterwards would materialise
    `[n_cells, V]` and keep it alive for backward -- ~115 MB per decoder at
    V=9,571 on a 3,000-node mini-batch, doubled for the two decoders and again
    at V=18,937.
    """
    if gene_ids is None:
        return lin(x)
    ids = gene_ids.reshape(-1)
    if lin.weight.shape[0] == ids.numel():
        # Same width already: the gather would be an identity permutation (a
        # sorted index set of size V over [0, V) can only be arange(V)), so skip
        # it and stay bit-identical to the single-panel path.
        return lin(x)
    bias = None if lin.bias is None else lin.bias[ids]
    return F.linear(x, lin.weight[ids], bias)


class MLP(MLP_Module):
    def __init__(
            self,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            hidden_channels: List[int] = [],
            dropout: float = 0.0,
            act: str = 'relu',
            norm: Optional[str] = None,
            plain_last: bool = True,
        ):
        """
        Initialize the MLP module.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - hidden_channels: List[int]
            The number of hidden channels representing the number of dimensions of the hidden features in the intermediate layers of the MLP.
        - dropout: float
            The dropout rate.
        - act: str
            The activation function.
        - norm: Optional[str]
            The normalization function.
        - plain_last: bool
            Whether to apply non-linearity, batch normalization and dropout to the last layer.

        Notes
        -----
        - If `in_channels` is not provided, the MLP module will assume the input channel is `mlp_params['hidden_channels'][0]`.
        - If `out_channels` is not provided, the MLP module will assume the output channel is `mlp_params['hidden_channels'][-1]`.
        """
        # if in_channels is provided, but out_channels is not, the MLP module will assume the output channel is `mlp_params['hidden_channels'][-1]`.
        if in_channels is not None and out_channels is None:
            channel_list = [in_channels] + hidden_channels
            
        # if out_channels is provided, but in_channels is not, the MLP module will assume the input channel is `mlp_params['hidden_channels'][0]`.
        elif in_channels is None and out_channels is not None:
            channel_list = hidden_channels + [out_channels]
        
        # if both in_channels and out_channels are provided, the MLP module will use them as the input and output channels.
        elif in_channels is not None and out_channels is not None:
            channel_list = [in_channels] + hidden_channels + [out_channels]

        # if both in_channels and out_channels are not provided, use hidden_channels as channel_list
        else:
            channel_list = hidden_channels

        assert len(channel_list) > 1, f"Channel list has length {len(channel_list)} which is less than 2. Please provide at least an input channel and an output channel."
        
        # initialize the MLP module using channel_list so that hidden layers of different dimensions can be used
        # e.g. 200 -> [400, 600] -> 1000 = 3 layers
        super().__init__(
            channel_list=channel_list,
            dropout=dropout,
            act=act,
            act_first=False,
            norm=norm,
            plain_last=plain_last,
        )
        

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: Optional[bool] = None,
        gene_ids: Optional[torch.Tensor] = None,
        out_gene_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of the
                final output layer. (default: :obj:`False`)
            gene_ids (torch.Tensor, optional): Cross-panel gather. When given,
                this MLP's FIRST layer was built at the gene-vocabulary width
                `V` while `x` is only `W` wide -- the union of the current
                block's panels -- so the layer's weight is indexed down to the
                columns the block actually carries:
                `F.linear(x, W_in[:, gene_ids], b_in)`.

                No separate module holds those weights: an `nn.Linear(V, h)`
                already IS the vocabulary-wide matrix, so indexing its weight
                is the whole mechanism, and autograd accumulates gradient into
                exactly those columns. Genes absent from a block therefore
                receive no update that step, which is correct -- they were not
                observed.

                `None` (the default) leaves every path byte-identical to the
                single-panel behaviour. (default: :obj:`None`)
            out_gene_ids (torch.Tensor, optional): The decoder-side mirror --
                gathers the LAST layer's output ROWS and bias instead of the
                first layer's input columns. Used by `MLPSoftmax`, whose
                vocabulary-wide layer is its output. (default: :obj:`None`)
        """
        # `return_emb` is annotated here as `NoneType` to be compatible with
        # TorchScript, which does not support different return types based on
        # the value of an input argument.
        emb: Optional[torch.Tensor] = None

        # If `plain_last=True`, then `len(norms) = len(lins) -1, thus skipping
        # the execution of the last layer inside the for-loop.
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = _lin_or_gather(lin, x, gene_ids if i == 0 else None)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            # `zip(self.lins, self.norms)` above stops one short of the last
            # layer when `plain_last`, so `lins[-1]` runs here. It is also
            # `lins[0]` when this MLP has a single layer, in which case the
            # loop did not execute and the gather belongs here instead.
            if len(self.lins) == 1 and gene_ids is not None:
                # One layer only: it is simultaneously first and last, so an
                # input gather belongs here too.
                x = _lin_or_gather(self.lins[-1], x, gene_ids)
            else:
                x = _lin_or_gather_out(self.lins[-1], x, out_gene_ids)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x