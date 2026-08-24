from typing import Optional, Literal

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from vqniche.modules import MLP as MLP_Module
from vqniche.modules import ConditionalMLP as ConditionalMLP_Module
from vqniche.modules import FiLM
from vqniche.modules import TemperatureAnnealer


class MLPSoftmax(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            apply_conditioning: Optional[Literal['pre-MLP', 'in-MLP']] = None,
            mlp_params: dict = {},
            conditioning_params: Optional[dict] = None,
            temperature_annealer_params: Optional[dict] = None,
        ):
        """
        Initialize the MLPSoftmax decoder.

        Parameters
        ----------
        - in_channels: int
            The number of input channels representing the number of dimensions of the latent embeddings from the encoder.
        - out_channels: int
            The number of output channels representing the number of dimensions of the input features to be reconstructed.
            
        - apply_conditioning: Optional[Literal['pre-MLP', 'in-MLP']]
            Whether to apply the FiLM conditioning to the latent embedding before the MLP module (pre-MLP) or within each layer of the MLP module (in-MLP).
        
        - mlp_params: dict
            MLP-related hyperparameters such as `hidden_channels` (number of hidden channels representing the number of dimensions of the hidden features in the intermediate layers of the MLP), `dropout` (dropout rate), `act` (activation function), and `norm` (normalization function).
            
        - conditioning_params: Optional[dict]
            Conditioning-related hyperparameters such as `condition_list` (list of condition names), `use_bias` (whether to use a bias term), `use_residual` (whether to use a residual connection), `residual_weight` (weight of the residual connection), and `init_mode` (initialization mode).
        
        - temperature_annealer_params: Optional[dict]
            Temperature annealing-related hyperparameters such as `temperature_start` (starting temperature), `temperature_end` (ending temperature), and `annealing_steps` (number of annealing steps).
        """
        super().__init__()

        # set parameters of the MLPSoftmax decoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_conditioning = apply_conditioning
        
        # no conditioning
        if apply_conditioning is None:
            self.mlp_module = MLP_Module(
                in_channels=in_channels,
                out_channels=out_channels,
                **mlp_params,
                plain_last=True,
            )

        # conditioning before the MLP module
        if apply_conditioning == 'pre-MLP':
            self.conditioning_module = FiLM(
                in_channels=in_channels,
                condition_dim=conditioning_params['condition_dim'],
                **conditioning_params,
            )
            self.mlp_module = MLP_Module(
                in_channels=in_channels,
                out_channels=out_channels,
                **mlp_params,
                plain_last=True,
            )

        # conditioning within the MLP module
        elif apply_conditioning == 'in-MLP':
            self.mlp_module = ConditionalMLP_Module(
                in_channels=in_channels,
                out_channels=out_channels,
                **mlp_params,
                plain_last=True,
                conditioning_params=conditioning_params,
                apply_to_last=False,
                film_position='post_norm',
            )

        # temperature annealer
        if temperature_annealer_params is not None:
            self.temperature_annealer = TemperatureAnnealer(
                **temperature_annealer_params,
            )
        else:
            self.temperature_annealer = None
            self.temperature = 1.0


    def forward(
            self,
            x: torch.Tensor,
            read_depth: torch.Tensor,
            conditions: Optional[torch.Tensor] = None,
            gene_ids: Optional[torch.Tensor] = None,
            gene_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Parameters
        ----------
        - x: torch.Tensor
            The input tensor.
            Dimensions: (batch_size, hidden_channels)
        - read_depth: torch.Tensor
            The read depth tensor.
            Dimensions: (batch_size, 1)
        - conditions: Optional[torch.Tensor]
            The conditions tensor.
            Dimensions: (batch_size, condition_dim)
        - gene_ids: Optional[torch.Tensor]
            Cross-panel gather on the OUTPUT layer. This decoder is built at the
            gene-vocabulary width `V`, but a block only carries `W` genes, so the
            final layer's weight ROWS and bias are indexed down to them.
            Dimensions: (1, W) or (W,)
        - gene_mask: Optional[torch.Tensor]
            Per-cell measured-gene mask, `True` where the cell's section
            measured that gene. REQUIRED whenever `gene_ids` is given, and the
            reason this decoder needed changing at all: the softmax below
            normalises across the gene axis, so without masking, probability
            mass leaks onto genes the section never measured and every measured
            prediction is systematically deflated. Nothing raises -- the
            reconstructions are just quietly wrong.
            Dimensions: (batch_size, W)

        Returns:
        -------
        - torch.Tensor:
            Output of the MLP followed by a softmax and a multiplication with the read depth.
        """
        # `out_gene_ids` narrows the OUTPUT layer to the block's genes. It is
        # the LAST layer that is vocabulary-wide here (the encoder's is its
        # first), and the gather is applied to that layer's weight rows rather
        # than to the emitted logits -- see `_lin_or_gather_out` for why that
        # distinction is worth ~115 MB per decoder at V=9,571.
        #
        # MLP without conditioning
        if self.apply_conditioning is None:
            xhat = self.mlp_module(x, out_gene_ids=gene_ids)
        
        # conditioning before MLP
        elif self.apply_conditioning == 'pre-MLP':
            assert conditions is not None, "conditions must be provided when applying conditioning before MLP"
            x = self.conditioning_module(
                x=x,
                conditions=conditions,
            )
            xhat = self.mlp_module(x, out_gene_ids=gene_ids)
        
        # conditioning within MLP
        elif self.apply_conditioning == 'in-MLP':
            assert conditions is not None, "conditions must be provided when applying conditioning within MLP"
            xhat = self.mlp_module(
                x=x,
                conditions=conditions,
                out_gene_ids=gene_ids,
            )
        
        # temperature annealing
        if self.temperature_annealer is not None:
            self.temperature = self.temperature_annealer.get_temp()
            self.temperature_annealer.step()
        # ---- masked softmax --------------------------------------------------
        # The softmax normalises ACROSS GENES, so with a cross-panel block the
        # unmeasured columns would absorb probability mass that belongs to the
        # measured ones, deflating every measured prediction. Masking the logits
        # to -inf removes those columns from the normaliser entirely, so the
        # measured genes' predictions sum to `read_depth` exactly -- which is
        # what makes the NB target and prediction comparable.
        if gene_mask is not None:
            if gene_mask.shape != xhat.shape:
                raise ValueError(
                    f"gene_mask {tuple(gene_mask.shape)} must match the decoder "
                    f"output {tuple(xhat.shape)}; a mismatch would mask the "
                    f"wrong genes."
                )
            # A row with nothing measured would make the whole softmax NaN.
            # Cannot arise (a section measures its own panel by construction),
            # so this is a guard against a future upstream bug, not a case to
            # handle silently.
            if not bool(gene_mask.any(dim=-1).all()):
                raise ValueError(
                    "gene_mask has a row with no measured gene; softmax over an "
                    "all -inf row is NaN."
                )
            xhat = xhat.masked_fill(~gene_mask, float("-inf"))

        xhat = F.softmax(xhat / self.temperature, dim=-1)

        # scale by empirical read depth
        xhat = xhat * read_depth.unsqueeze(-1)

        return xhat
