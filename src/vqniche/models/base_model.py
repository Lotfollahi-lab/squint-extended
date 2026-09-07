import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Literal

import pandas as pd

import torch
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.nn.dense.linear import Linear
import torch.nn as nn
import torch.nn.functional as F

from vqniche.modules.mlp import MLP as MLP_AdjacencyDecoder
from ..decoders.mlp_softmax import MLPSoftmax
from vqniche.loss import (
    cross_entropy_loss,
    mse_attribute_reconstruction_loss,
    nb_attribute_reconstruction_loss,
    nb_nbr_attribute_reconstruction_loss,
    nb_nbr_attribute_reconstruction_loss_dual,
    contrastive_cell_attribute_loss,
    contrastive_cell_attribute_within_batch_loss,
    contrastive_cell_attribute_cross_batch_mnn_loss,
    disentangle_cell_niche_loss,
    cell_niche_alignment_loss,
    mse_adjacency_reconstruction_loss,
    bce_adjacency_reconstruction_loss,
    bce_cosine_adjacency_reconstruction_loss,
    adversarial_batch_loss,
    mmd_batch_loss,
    mmd_prior_loss,
    mse_joint_code_commit_loss,
    ce_spatial_prior_loss,
    mse_commit_loss,
    mse_code_loss,
    mse_commit_loss_cell,
    mse_commit_loss_niche,
    l2_codebook_orthogonal_regularization_loss,
    mask_token_regularization
)
from vqniche.utils.type_conversions import inference_data_dict_to_adata
from vqniche.metrics import compute_benchmarking_metrics


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            encoder_name: str,
            attribute_decoder_name: str,
            adjacency_decoder_name: str,
            predictor_name: str,
            in_channels: int = None,
            out_channels: int = None,
            train_metrics_list: List[str] = [],
            test_metrics_list: List[str] = [],
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
            mask_lr_scale: float = 1.0,
            fused: bool = False,
            no_decay_patterns: Optional[List[str]] = None,
            lr_schedule: Literal['none', 'cosine'] = 'none',
            warmup_steps: int = 0,
            min_lr_ratio: float = 0.0,
            total_steps: Optional[int] = None,
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'mean'},
        ) -> None:
        """
        Initialize the BaseModel class.

        Parameters
        ----------
        - model_name: str
            The name of the model.
        - encoder_name: str
            The encoder name.
        - attribute_decoder_name: str
            The name of the attribute decoder module.
        - adjacency_decoder_name: str
            The name of the adjacency decoder module.
        - predictor_name: str
            The name of the predictor module.

        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.

        - train_metrics_list: List[str]
            The list of metrics to compute during training.
        - test_metrics_list: List[str]
            The list of metrics to compute during testing.

        - optimizer_name: str
            The optimizer name.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.
        - mask_lr_scale: float
            The learning rate scale for the learnable mask.
        - no_decay_patterns: Optional[List[str]]
            Parameter-name substrings to EXEMPT from weight decay. None
            (default) keeps the legacy single-group optimizer.
        - lr_schedule: 'none' | 'cosine'
            Learning-rate schedule. 'none' (default) keeps the legacy
            constant-LR behaviour and returns a bare optimizer.
        - warmup_steps: int
            Linear warmup length in OPTIMIZER STEPS. Ignored when
            `lr_schedule='none'`.
        - min_lr_ratio: float
            Floor of the cosine decay, as a fraction of `lr`.
        - total_steps: Optional[int]
            Schedule horizon in optimizer steps. When None it is taken
            from `trainer.estimated_stepping_batches`.

        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            The loss function keyword arguments.
        """
        # names of the model components
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.attribute_decoder_name = attribute_decoder_name
        self.adjacency_decoder_name = adjacency_decoder_name
        self.predictor_name = predictor_name

        super().__init__()

        # Data parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # metrics to compute during training
        self.train_metrics_list = train_metrics_list
        self.test_metrics_list = test_metrics_list

        self.on_train_epoch_end_logs_df = pd.DataFrame()

        # Optimizer parameters
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.mask_lr_scale = mask_lr_scale
        # `fused=True` on Adam coalesces all parameter updates into a
        # single CUDA kernel (vs per-tensor launches). 5-15% speedup on
        # the optimizer step for SQUINT's many-small-tensor regime.
        # Requires all params on GPU + bf16-mixed master weights stay
        # fp32, both of which we satisfy. Default False to preserve
        # legacy behaviour for any caller that builds a cfg without
        # the new key; set via `cfg["model"]["optimizer_params"]["fused"]`.
        self.fused = bool(fused)

        # LR schedule. Default 'none' returns a bare optimizer, which is
        # exactly the legacy path -- every variant predating this keyword
        # trains as it did before.
        # Parameters exempted from weight decay, by name substring.
        #
        # WHY THIS EXISTS. Adam applies weight decay every step regardless of
        # whether a parameter received gradient, and the adaptive
        # normalisation makes the decay step ~`lr` no matter the magnitude --
        # so a zero-gradient parameter reaches EXACTLY 0.0 within ~1,000
        # steps at lr=7e-4, wd=1e-3. Measured on the corpus runs: the
        # per-batch `batch_embedding` had only 52/416, 69/416 and 67/416
        # non-zero rows, because a section appears in roughly one block out
        # of ~114 and then takes ~198,000 steps of pure decay. The decoder
        # covariate -- the paper's batch-correction lever -- was structurally
        # disabled for ~85% of sections in every run.
        #
        # Embeddings are conventionally excluded from weight decay for
        # exactly this reason. Default None = legacy single group, so every
        # variant predating this trains as before.
        self.no_decay_patterns = list(no_decay_patterns or [])
        self.lr_schedule = str(lr_schedule)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.total_steps = None if total_steps is None else int(total_steps)
        # Resolved in `configure_optimizers` when a schedule is active.
        self._horizon = None

        # Loss parameters
        self.loss_kwargs = loss_kwargs

        # Dispersion parameter for the NB likelihood.
        # When `code_conditional_dispersion` is enabled (set later by the
        # subclass after the encoder is built), we still keep this scalar
        # parameter as a fallback for any code path that needs a global
        # dispersion (e.g. inference paths that don't have a code yet).
        # Subclasses that enable code-conditional dispersion will use
        # `self.dispersion_head(h_quantized)` in lieu of `torch.exp(self.dispersion)`
        # when assembling `loss_data['dispersion']`.
        self.dispersion = torch.nn.Parameter(torch.randn(self.in_channels))

        # Whether the NB dispersion is conditioned on the quantized embedding
        # (a small MLP head: z_q -> log(theta)). Default off; flipped on by
        # the VQNiche subclass when `code_conditional_dispersion=True`.
        self.code_conditional_dispersion = False
        self.dispersion_head = None

        self.loss_fn_tuples = self.set_loss_fn_tuples(loss_names, loss_kwargs)

        self.save_hyperparameters()


    def set_loss_fn_tuples(
            self,
            loss_fn_names: List[str],
            loss_kwargs: dict = {}
        ) -> List[Tuple[str, Callable, List[str], dict]]:
        """
        Set the loss functions for the model.

        Parameters
        ----------
        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        loss_fn_tuples: List
            One tuple per loss name in loss_names comprising of loss function name (str), loss function (callable), list of data related key strings required to be passed to the loss function, and a dictionary of additional keyword arguments for the loss function.

        Notes
        -----
        Use this method to set the loss functions for the encoder. The loss functions should be defined in the utils.loss module.
        """
        # initialize a list to store the loss function tuples
        loss_fn_tuples = []

        print("Setting the following loss terms as criterion for training:")
        for loss_fn_name in loss_fn_names:
            print(f"Loss function: {loss_fn_name}")
            loss_fn_params = {}

            if loss_fn_name == 'cross_entropy_loss':
                # set the cross-entropy loss function
                loss_fn = cross_entropy_loss

                # set key names for data required to compute cross-entropy loss
                loss_fn_data_keys = ['logits', 'labels']

                # set keyword parameters for cross-entropy loss
                wt_cross_entropy = loss_kwargs.get('wt_cross_entropy')
                if wt_cross_entropy is not None:
                    loss_fn_params['wt_cross_entropy'] = wt_cross_entropy

            elif loss_fn_name == 'mse_attribute_reconstruction_loss':
                loss_fn = mse_attribute_reconstruction_loss

                loss_fn_data_keys = ['pred_attr', 'target_attr']

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'nb_attribute_reconstruction_loss':
                loss_fn = nb_attribute_reconstruction_loss

                # `gene_mask` is the cross-panel measured-gene mask, and all
                # three NB entries request it unconditionally: `_loss_fn_data`
                # builds its dict by key lookup, so an absent key raises
                # KeyError. `loss_data` supplies None on the single-panel path
                # and the loss reads that as "no mask", leaving its reduction
                # byte-identical to before.
                loss_fn_data_keys = ['pred_attr', 'target_attr', 'edge_index', 'batch_size', 'dispersion',
                                     'gene_mask']

                k_hop_nb_loss = loss_kwargs.get('k_hop_nb_loss')
                if k_hop_nb_loss is not None:
                    loss_fn_params['k_hop_nb_loss'] = k_hop_nb_loss

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'nb_attribute_reconstruction_loss_nbr':
                # Second NB reconstruction term for recon_mode='both'.
                # Uses nb_nbr_attribute_reconstruction_loss — a thin wrapper
                # that accepts pred_attr_nbr / target_attr_nbr kwargs so the
                # dispatcher's key-based lookup doesn't collide with the cell
                # branch's pred_attr / target_attr keys.
                loss_fn = nb_nbr_attribute_reconstruction_loss

                loss_fn_data_keys = ['pred_attr_nbr', 'target_attr_nbr', 'edge_index', 'batch_size', 'dispersion',
                                     'gene_mask']

                # k_hop_nb_loss=0: aggregation was done upstream in
                # training_step; the wrapper must NOT re-aggregate.
                loss_fn_params['k_hop_nb_loss'] = 0

                wt_attr_reconstr_nbr = loss_kwargs.get('wt_attr_reconstr_nbr')
                if wt_attr_reconstr_nbr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr_nbr
                else:
                    wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                    if wt_attr_reconstr is not None:
                        loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'mse_adjacency_reconstruction_loss':
                loss_fn = mse_adjacency_reconstruction_loss

                loss_fn_data_keys = ['batch_size', 'h_adj', 'batch_edge_index']

                estimate_adj_kwargs = loss_kwargs.get('estimate_adj_kwargs')
                if estimate_adj_kwargs is not None:
                    loss_fn_params['estimate_adj_kwargs'] = estimate_adj_kwargs

                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'bce_adjacency_reconstruction_loss':
                loss_fn = bce_adjacency_reconstruction_loss

                loss_fn_data_keys = ['batch_size', 'h_adj', 'batch_edge_index']

                # Within-section pair scope. Default True: cross-section
                # pairs are dropped from the BCE so two biologically-
                # similar cells from different AnnData sections aren't
                # pushed apart by the spatial-adjacency loss (which was
                # the legacy global-pair behaviour). Set
                # `loss_kwargs['adj_within_section_only'] = False` to
                # revert to legacy. Requires the model's `_step` to put a
                # per-node `node_adata_batch_ids` tensor in the loss data.
                if loss_kwargs.get('adj_within_section_only', True):
                    loss_fn_data_keys.append('node_adata_batch_ids')

                edge_sampling_ratio = loss_kwargs.get('edge_sampling_ratio')
                if edge_sampling_ratio is not None:
                    loss_fn_params['edge_sampling_ratio'] = edge_sampling_ratio

                use_pos_weight = loss_kwargs.get('use_pos_weight')
                if use_pos_weight is not None:
                    loss_fn_params['use_pos_weight'] = use_pos_weight

                estimate_adj_kwargs = loss_kwargs.get('estimate_adj_kwargs')
                if estimate_adj_kwargs is not None:
                    loss_fn_params['estimate_adj_kwargs'] = estimate_adj_kwargs

                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'nb_attribute_reconstruction_loss_nbr_dual':
                # Niche-branch NB for VQNiche_Dual with a SEPARATE dispersion
                # parameter (`dispersion_niche`) decoupled from the cell-branch
                # dispersion. See `nb_nbr_attribute_reconstruction_loss_dual`
                # for the rationale (variance structure differs between
                # per-cell and neighbourhood-mean targets).
                loss_fn = nb_nbr_attribute_reconstruction_loss_dual
                loss_fn_data_keys = ['pred_attr_nbr', 'target_attr_nbr',
                                     'edge_index', 'batch_size', 'dispersion_niche',
                                     'gene_mask']
                loss_fn_params['k_hop_nb_loss'] = 0
                wt_attr_reconstr_nbr = loss_kwargs.get('wt_attr_reconstr_nbr')
                if wt_attr_reconstr_nbr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr_nbr
                else:
                    wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                    if wt_attr_reconstr is not None:
                        loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'contrastive_cell_attribute_loss':
                # NT-Xent contrastive auxiliary loss on the pre-quantization
                # cell-branch latent (`quantizer_input_cell` = z_mlp_cell of
                # the seed cells). Positive pairs are the top-k cells with
                # the most similar log1p gene-expression profile in the
                # FULL mini-batch (any section); negatives are the rest.
                # Pulls same-type cells together in `z_mlp_cell` space and
                # pushes different-type apart — a between-cell objective
                # that complements NB's within-cell objective. Tunables
                # come from loss_kwargs; all are optional with sensible
                # defaults in the loss fn.
                loss_fn = contrastive_cell_attribute_loss
                loss_fn_data_keys = ['quantizer_input_cell', 'target_attr']
                for k in (
                    'k_pos',
                    'temperature',
                    'log_transform_gene_space',
                    'wt_contrastive_cell',
                ):
                    v = loss_kwargs.get(k)
                    if v is not None:
                        loss_fn_params[k] = v

            elif loss_fn_name == 'contrastive_cell_attribute_within_batch_loss':
                # Within-section variant of the contrastive cell loss. Both
                # positive-pair candidates AND the NT-Xent denominator are
                # restricted to cells sharing the anchor's adata_batch_id.
                # Same architectural reasoning as
                # `adj_within_section_only=True` on the cosine adjacency
                # BCE: without the restriction, cross-section pairs leak
                # into the gradient and push biologically-similar cells
                # from different MERFISH/STARmap sections apart — counter
                # to the batch-integration objective.
                loss_fn = contrastive_cell_attribute_within_batch_loss
                loss_fn_data_keys = [
                    'quantizer_input_cell', 'target_attr',
                    'node_adata_batch_ids', 'batch_size',
                ]
                for k in (
                    'k_pos',
                    'temperature',
                    'log_transform_gene_space',
                    'wt_contrastive_cell',
                ):
                    v = loss_kwargs.get(k)
                    if v is not None:
                        loss_fn_params[k] = v

            elif loss_fn_name == 'contrastive_cell_attribute_cross_batch_mnn_loss':
                # Within-batch NT-Xent (resolution) + cross-batch mutual-NN
                # pure-attraction term (integration). wt_cross=0 reduces to the
                # within-batch loss. Needs the same per-node section ids.
                loss_fn = contrastive_cell_attribute_cross_batch_mnn_loss
                loss_fn_data_keys = [
                    'quantizer_input_cell', 'target_attr',
                    'node_adata_batch_ids', 'batch_size',
                ]
                for k in (
                    'k_pos',
                    'k_cross',
                    'temperature',
                    'log_transform_gene_space',
                    'wt_contrastive_cell',
                    'wt_cross',
                    'mnn_floor',
                    'mutual',
                ):
                    v = loss_kwargs.get(k)
                    if v is not None:
                        loss_fn_params[k] = v

            elif loss_fn_name == 'disentangle_cell_niche_loss':
                # Decorrelation penalty between the cell and niche latents
                # (Barlow-Twins-style cross-correlation). Pushes the niche
                # code to carry only signal complementary to the cell code.
                # Reads the two pre-VQ branch latents already sliced to the
                # seed cells (z_mlp[:bs] / z_gnn[:bs]).
                loss_fn = disentangle_cell_niche_loss
                loss_fn_data_keys = [
                    'quantizer_input_cell', 'quantizer_input_niche',
                ]
                for k in ('wt_disentangle',):
                    v = loss_kwargs.get(k)
                    if v is not None:
                        loss_fn_params[k] = v

            elif loss_fn_name == 'cell_niche_alignment_loss':
                # Cross-branch ALIGNMENT (Barlow invariance term): pushes the
                # matched cell/niche feature dims to be correlated — the
                # opposite sign of the disentanglement penalty. Same two
                # pre-VQ branch latents.
                loss_fn = cell_niche_alignment_loss
                loss_fn_data_keys = [
                    'quantizer_input_cell', 'quantizer_input_niche',
                ]
                for k in ('wt_align',):
                    v = loss_kwargs.get(k)
                    if v is not None:
                        loss_fn_params[k] = v

            elif loss_fn_name == 'bce_cosine_adjacency_reconstruction_loss':
                # NicheCompass-style adjacency reconstruction via cosine
                # similarity. Operates on either:
                #   - 'z_gnn' (continuous, default; NicheCompass-faithful)
                #   - 'z_q_niche' (quantized, opt-in)
                # Selected by `loss_kwargs['adj_loss_input']`.
                loss_fn = bce_cosine_adjacency_reconstruction_loss

                adj_input = loss_kwargs.get('adj_loss_input', 'z_gnn')
                if adj_input not in ('z_gnn', 'z_q_niche'):
                    raise ValueError(
                        f"loss_kwargs['adj_loss_input'] must be one of "
                        f"{{'z_gnn', 'z_q_niche'}}, got {adj_input!r}."
                    )
                loss_fn_data_keys = ['batch_size', 'batch_edge_index', adj_input]

                # Within-section pair scope (default True). See the
                # `bce_adjacency_reconstruction_loss` branch above for
                # rationale + opt-out flag.
                if loss_kwargs.get('adj_within_section_only', True):
                    loss_fn_data_keys.append('node_adata_batch_ids')

                edge_sampling_ratio = loss_kwargs.get('edge_sampling_ratio')
                if edge_sampling_ratio is not None:
                    loss_fn_params['edge_sampling_ratio'] = edge_sampling_ratio
                use_pos_weight = loss_kwargs.get('use_pos_weight')
                if use_pos_weight is not None:
                    loss_fn_params['use_pos_weight'] = use_pos_weight
                cosine_temperature = loss_kwargs.get('cosine_temperature')
                if cosine_temperature is not None:
                    loss_fn_params['cosine_temperature'] = cosine_temperature
                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'adversarial_batch_loss':
                # Domain-adversarial batch invariance for VQNiche_Dual.
                # CE on the BatchAdversaryHead's logits against per-cell
                # batch IDs. The head's GRL flips the gradient sign so
                # the encoder is pushed toward batch-invariance while the
                # classifier itself trains normally.
                loss_fn = adversarial_batch_loss
                loss_fn_data_keys = ['batch_logits', 'batch_labels']
                wt_adv_batch = loss_kwargs.get('wt_adv_batch')
                if wt_adv_batch is not None:
                    loss_fn_params['wt_adv_batch'] = wt_adv_batch

            elif loss_fn_name == 'mmd_batch_loss':
                # Non-adversarial batch-invariance loss. Computes
                # differentiable MMD between the per-batch distributions
                # of `mmd_target` (typically z_mlp[:batch_size], the
                # cell-token input pre-VQ) and adds it to the total
                # loss. Unlike the adversarial CE, MMD has no min-max
                # game — backprop directly minimises the kernel-based
                # distribution distance, so it doesn't suffer the
                # warmup pathology where the cell token learns
                # batch-correlated features early and the late-arriving
                # adversary can't remove them.
                #
                # We read `mmd_target_labels` (NOT `batch_labels`)
                # because the adversarial CE may also be active and
                # populate `batch_labels` with the FULL-tensor variant
                # (seeds + sampled neighbours), whereas MMD always
                # operates on the seed prefix only — so it needs the
                # seed-only label slice.
                loss_fn = mmd_batch_loss
                # `mmd_group_labels` SCOPES the pairwise sum, and omitting it
                # here is not a no-op -- it silently runs the loss GLOBALLY.
                # At corpus scale that aligns cells across organs (18 training
                # tissues, >1 tissue in 100% of blocks) and deletes the
                # strongest signal in the code space. `_step` always populates
                # the key, using None when no group map is configured, so this
                # list can request it unconditionally.
                loss_fn_data_keys = ['mmd_target', 'mmd_target_labels',
                                     'mmd_group_labels']
                wt_mmd_batch = loss_kwargs.get('wt_mmd_batch')
                if wt_mmd_batch is not None:
                    loss_fn_params['wt_mmd_batch'] = wt_mmd_batch
                mmd_n_sub = loss_kwargs.get('mmd_n_sub')
                if mmd_n_sub is not None:
                    loss_fn_params['n_sub'] = int(mmd_n_sub)

            elif loss_fn_name == 'mmd_prior_loss':
                # MMD between mmd_target and samples from the
                # isotropic Gaussian prior N(0, prior_std² * I).
                # Parameterless analogue of a VAE's KL(q || N(0, I))
                # — pulls the empirical distribution of the embedding
                # toward a batch-agnostic prior, providing
                # NicheCompass-style integration pressure without
                # making the encoder probabilistic. Consumes the
                # same `mmd_target` tensor as `mmd_batch_loss` (the
                # two can coexist; they consume the same tensor but
                # compute different distances). No batch labels
                # needed.
                loss_fn = mmd_prior_loss
                loss_fn_data_keys = ['mmd_target']
                wt_mmd_prior = loss_kwargs.get('wt_mmd_prior')
                if wt_mmd_prior is not None:
                    loss_fn_params['wt_mmd_prior'] = wt_mmd_prior
                mmd_n_sub = loss_kwargs.get('mmd_n_sub')
                if mmd_n_sub is not None:
                    loss_fn_params['n_sub'] = int(mmd_n_sub)
                mmd_prior_std = loss_kwargs.get('mmd_prior_std')
                if mmd_prior_std is not None:
                    loss_fn_params['prior_std'] = float(mmd_prior_std)

            elif loss_fn_name == 'mse_commit_loss_cell':
                # Commit loss for the CELL branch of VQNiche_Dual. Pulls
                # z_mlp toward z_q_cell. Disjoint from the niche branch.
                loss_fn = mse_commit_loss_cell
                loss_fn_data_keys = ['quantizer_input_cell', 'quantizer_output_cell']
                wt_commit_cell = loss_kwargs.get('wt_commit_cell')
                if wt_commit_cell is None:
                    wt_commit_cell = loss_kwargs.get('wt_commit')
                if wt_commit_cell is not None:
                    loss_fn_params['wt_commit'] = wt_commit_cell

            elif loss_fn_name == 'mse_commit_loss_niche':
                # Commit loss for the NICHE branch of VQNiche_Dual. Pulls
                # z_gnn toward z_q_niche. Disjoint from the cell branch.
                loss_fn = mse_commit_loss_niche
                loss_fn_data_keys = ['quantizer_input_niche', 'quantizer_output_niche']
                wt_commit_niche = loss_kwargs.get('wt_commit_niche')
                if wt_commit_niche is None:
                    wt_commit_niche = loss_kwargs.get('wt_commit')
                if wt_commit_niche is not None:
                    loss_fn_params['wt_commit'] = wt_commit_niche

            elif loss_fn_name == 'mse_joint_code_commit_loss':
                loss_fn = mse_joint_code_commit_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_joint_code_commit = loss_kwargs.get('wt_joint_code_commit')
                if wt_joint_code_commit is not None:
                    loss_fn_params['wt_joint_code_commit'] = wt_joint_code_commit
                    
            elif loss_fn_name == 'ce_spatial_prior_loss':
                loss_fn = ce_spatial_prior_loss

                loss_fn_data_keys = ['h_spatial_prior', 'indices_one_hot']

                wt_spatial_prior = loss_kwargs.get('wt_spatial_prior')
                if wt_spatial_prior is not None:
                    loss_fn_params['wt_spatial_prior'] = wt_spatial_prior

            elif loss_fn_name == 'mse_commit_loss':
                loss_fn = mse_commit_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_commit = loss_kwargs.get('wt_commit')
                if wt_commit is not None:
                    loss_fn_params['wt_commit'] = wt_commit

            elif loss_fn_name == 'mse_code_loss':
                loss_fn = mse_code_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_code = loss_kwargs.get('wt_code')
                if wt_code is not None:
                    loss_fn_params['wt_code'] = wt_code

            elif loss_fn_name == 'l2_codebook_orthogonal_regularization_loss':
                loss_fn = l2_codebook_orthogonal_regularization_loss

                loss_fn_data_keys = ['codebook_embeddings']

                wt_codebook_orthogonal_regularization = loss_kwargs.get('wt_codebook_orthogonal_regularization')
                if wt_codebook_orthogonal_regularization is not None:
                    loss_fn_params['wt_codebook_orthogonal_regularization'] = wt_codebook_orthogonal_regularization

                codebook_reg_active_codes_only = loss_kwargs.get('codebook_reg_active_codes_only')
                if codebook_reg_active_codes_only is not None:
                    loss_fn_params['codebook_reg_active_codes_only'] = codebook_reg_active_codes_only

                codebook_reg_max_codes = loss_kwargs.get('codebook_reg_max_codes')
                if codebook_reg_max_codes is not None:
                    loss_fn_params['codebook_reg_max_codes'] = codebook_reg_max_codes

            elif loss_fn_name == 'mask_token_regularization':
                loss_fn = mask_token_regularization

                loss_fn_data_keys = ['mask_token']

                wt_mask_token_regularization = loss_kwargs.get('wt_mask_token_regularization')
                if wt_mask_token_regularization is not None:
                    loss_fn_params['wt_mask_token_regularization'] = wt_mask_token_regularization

            else:
                raise NotImplementedError(f'{loss_fn_name} Loss not implemented')

            loss_fn_tuple = (loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params)
            loss_fn_tuples.append(loss_fn_tuple)

        return loss_fn_tuples


    def criterion(
            self,
            loss_data: dict,
            curr_batch_size: Optional[int] = None,
            mode: Literal['train', 'val'] = 'train',
        ) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters
        ----------
        - loss_data: dict
            A collection of data objects required to compute the loss.
        - curr_batch_size: int
            The number of samples in the current batch. Required for logging.
        - mode: Literal['train', 'val']
            The mode of the model (train, val).

        Returns
        -------
        - total_loss: torch.Tensor
            The total computed loss across all loss terms for the current batch.
        """
        assert len(self.loss_fn_tuples) > 0, 'No loss functions defined'

        # initialize total_loss = 0.0 with requires_grad=True so that the loss can be backpropagated
        # total_loss will be computed as the sum of all the loss terms from self.loss_names
        total_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float32).to(self.device)
        # Accumulators for the reconstruction-vs-graph balance diagnostic.
        _nb_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        _adj_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # during model initialization, self.loss_fn_tuples is set to a list of tuples
        # one tuple per loss name in self.loss_names
        # each tuple contains the loss function name, the callable loss function, the data keys required to compute the loss, and the loss function parameters
        for loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params in self.loss_fn_tuples:
            # extract from loss_data, the data required to compute the current loss function
            _loss_fn_data = {key: loss_data[key] for key in loss_fn_data_keys}

            # pass the extracted data to the loss function along with the loss function related kwargs
            loss_fn_value = loss_fn(**_loss_fn_data, **loss_fn_params)
            # A loss that falls off the end of a branch returns None, and the
            # bare `torch.add` below then raises "argument 'other' must be
            # Tensor, not NoneType" -- naming neither the loss nor the inputs.
            # That cost a debugging cycle on the MMD sweep, so say which one.
            if not isinstance(loss_fn_value, torch.Tensor):
                # Identify the callable itself. A function whose every exit
                # returns a Tensor returned None here, and its own end-of-body
                # guard did not fire -- which can only mean the object being
                # called is not the source being read. Report where it lives.
                import inspect as _inspect
                try:
                    _src_file = _inspect.getsourcefile(loss_fn)
                    _src = _inspect.getsource(loss_fn)
                    _nret = _src.count("return ")
                    _guard = "produced" in _src
                except (OSError, TypeError):
                    _src_file, _nret, _guard = "<unavailable>", -1, False
                raise TypeError(
                    f"loss {loss_fn_name!r} returned "
                    f"{type(loss_fn_value).__name__}, not a Tensor. Data keys "
                    f"{loss_fn_data_keys} had shapes "
                    f"{ {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in _loss_fn_data.items()} }"
                    f", params {loss_fn_params}. "
                    f"callable={getattr(loss_fn, '__module__', '?')}."
                    f"{getattr(loss_fn, '__qualname__', '?')} "
                    f"file={_src_file} returns={_nret} has_guard={_guard}"
                )
            # add the computed loss to the total_loss
            total_loss = torch.add(total_loss, loss_fn_value)

            # log each computed loss term
            self.log(
                    name=f"{mode}_{loss_fn_name}",
                    value=loss_fn_value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=curr_batch_size,
                    sync_dist=True,
                    )

            # Realised contribution of the reconstruction vs the graph term,
            # accumulated for the balance diagnostic below. `loss_fn_value` is
            # ALREADY weighted -- `wt_adj_reconstr` is a loss_fn_param applied
            # inside the adjacency loss -- so these are what the optimizer
            # actually sees, not the raw terms.
            if "nb_attribute_reconstruction" in loss_fn_name:
                _nb_sum = _nb_sum + loss_fn_value.detach()
            elif "adjacency_reconstruction" in loss_fn_name:
                _adj_sum = _adj_sum + loss_fn_value.detach()

            # free up memory by deleting the intermediate loss function data
            del _loss_fn_data

        self._log_recon_graph_balance(
            _nb_sum, _adj_sum, loss_data, mode, curr_batch_size,
        )
        return total_loss

    def _log_recon_graph_balance(self, nb_sum, adj_sum, loss_data, mode,
                                 curr_batch_size) -> None:
        """
        Log the realised NB:adjacency balance and the panel width it occurred at.

        WHY. The NB term is `mean over cells of SUM over genes`
        (nb_attribute_reconstruction.py:128), so it grows with the number of
        MEASURED genes. The adjacency BCE is over edge pairs and does not. Across
        a corpus spanning 169 to 5,049 genes that is a ~30x swing in the balance
        between reconstructing expression and reconstructing structure — from a
        single `wt_adj_reconstr`. That constant was tuned on the MERFISH/STARmap
        MB panel (~950 genes; run_squint.py:2117-2127, "~150 nats" of NB against
        an adjacency BCE bounded by log(2)), so the corpus sits mostly ABOVE its
        tuning width: ~5.3x the NB magnitude at 5,049 genes, ~5.6x below it at
        169.

        Whether that is a bug is genuinely open. A narrow panel carries less
        expression information while its spatial graph is just as informative, so
        relatively up-weighting the graph term for it may well be CORRECT. That
        is an empirical question, and answering it needs the realised ratio
        alongside the width it was measured at — which is what this logs, rather
        than pre-emptively inventing a width-dependent weight.

        `panel_width_mean` is per-cell measured genes from `gene_mask` under
        cross-panel, where a block mixes panels and a single width would be
        meaningless; it falls back to the input width otherwise.
        """
        if adj_sum is None or not torch.is_tensor(adj_sum) or float(adj_sum) == 0.0:
            return
        log = lambda n, v: self.log(  # noqa: E731 - local, one shape
            name=n, value=v, prog_bar=False, on_step=False, on_epoch=True,
            batch_size=curr_batch_size, sync_dist=True,
        )
        log(f"{mode}_recon_over_graph", nb_sum / adj_sum)

        gene_mask = loss_data.get("gene_mask", None)
        if gene_mask is not None and torch.is_tensor(gene_mask):
            width = gene_mask.sum(dim=-1).to(torch.float32).mean()
        else:
            x = loss_data.get("x", None)
            if x is None or not torch.is_tensor(x) or x.dim() != 2:
                return
            width = torch.tensor(float(x.shape[1]), device=nb_sum.device)
        log(f"{mode}_panel_width_mean", width)


    def _init_attribute_decoder(
            self,
            in_channels: int,
            out_channels: int,
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            attribute_decoder_params: dict = {},
        ) -> pl.LightningModule:
        """
        Initialize the attribute decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the attribute decoder module.
        - out_channels: int
            The output dimension of the attribute decoder module.
        - attribute_decoder_name: Literal['MLPSoftmax']
            The name of the attribute decoder module.
        - attribute_decoder_params: dict
            The parameters for the attribute decoder module.

        Returns
        -------
        - attribute_decoder: pl.LightningModule
            The attribute decoder module.
        """
        if attribute_decoder_name == 'MLPSoftmax':
            return MLPSoftmax(
                in_channels=in_channels,
                out_channels=out_channels,
                **attribute_decoder_params,
            )


    def _init_adjacency_decoder(
            self,
            in_channels: int,
            out_channels: int = 600,
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            mlp_params: dict = {},
        ) -> torch.nn.Module:
        """
        Initialize the adjacency decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the adjacency decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - mlp_params: dict
            The parameters for the MLP module.
        - conditioning_params: dict
            The parameters for the conditioning module.

        Returns
        -------
        - adjacency_decoder: torch.nn.Module
            The adjacency decoder module.
        """
        if adjacency_decoder_name == 'MLP_AdjacencyDecoder':
            return MLP_AdjacencyDecoder(
                in_channels=in_channels,
                out_channels=out_channels,
                **mlp_params,
            )


    def _init_predictor(
            self,
            predictor_name: Literal['Linear'] = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            init_method: str = 'kaiming_uniform'
        ) -> pl.LightningModule:
        """
        Initialize the predictor module.

        Parameters
        ----------
        - predictor_name: str
            The name of the predictor module.
        - in_channels: int
            The input dimension of the predictor module.
        - out_channels: int
            The output dimension of the predictor module.
        - init_method: str
            The initialization method for the predictor module.

        Returns
        -------
        - predictor: pl.LightningModule
            The predictor module.
        """
        if predictor_name == 'Linear':
            return Linear(
                in_channels=in_channels,
                out_channels=out_channels,
                weight_initializer=init_method
            )


    def _schedule_horizon(self) -> int:
        """
        Total optimizer steps the LR schedule should span.

        Prefers the explicit `total_steps` because
        `trainer.estimated_stepping_batches` is unreliable on the
        streaming backend: a corpus epoch is ~187,562 steps and the
        whole budget finishes inside epoch 0, so Lightning's estimate
        comes from `max_steps` when set and from a per-epoch batch
        count that the block loader cannot report otherwise.

        Returns
        -------
        - int
            The horizon, always >= 1.
        """
        if self.total_steps is not None:
            return max(1, self.total_steps)

        est = getattr(self.trainer, "estimated_stepping_batches", None) if self.trainer else None
        if est is None or not math.isfinite(float(est)) or float(est) <= 0:
            raise ValueError(
                "lr_schedule='cosine' needs a finite step horizon, but "
                f"trainer.estimated_stepping_batches is {est!r}. Set "
                "`optimizer_params.total_steps` (normally equal to "
                "`trainer.max_steps`) explicitly."
            )
        return max(1, int(est))

    def _lr_lambda(self, step: int) -> float:
        """
        The LR multiplier at a given optimizer step.

        Linear warmup, then cosine decay to `min_lr_ratio`. `step` is
        0-based, and the multiplier is clamped at the floor past the
        horizon rather than being allowed to turn back up.
        """
        warmup = self.warmup_steps
        if step < warmup:
            # (step + 1) so the very first step is not a zero-LR no-op.
            return (step + 1) / max(1, warmup)
        horizon = self._horizon
        prog = (step - warmup) / max(1, horizon - warmup)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos

    def configure_optimizers(self):
        """
        Configure the optimizer, and the LR scheduler when one is asked for.

        Returns
        -------
        - torch.optim.Optimizer, or dict
            A bare optimizer when `lr_schedule='none'` (the legacy
            behaviour, preserved so pre-existing variants are
            unaffected); otherwise Lightning's
            `{"optimizer": ..., "lr_scheduler": ...}` mapping stepped
            per optimizer step.
        """
        optimizer = self._build_optimizer()

        if self.lr_schedule == 'none':
            return optimizer
        if self.lr_schedule != 'cosine':
            raise NotImplementedError(
                f"lr_schedule {self.lr_schedule!r} not implemented "
                "(expected 'none' or 'cosine')"
            )

        # Resolved once, here rather than inside the lambda: the horizon
        # must not change under the schedule mid-run, and reading the
        # trainer on every step would.
        self._horizon = self._schedule_horizon()
        if self.warmup_steps >= self._horizon:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than the "
                f"schedule horizon ({self._horizon}); the LR would never decay."
            )
        print(
            f"LR schedule: cosine, base lr={self.lr:g}, "
            f"warmup={self.warmup_steps} steps, horizon={self._horizon} steps, "
            f"floor={self.min_lr_ratio:g}x"
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _param_groups(self):
        """
        Optimizer parameter groups, splitting out anything exempt from decay.

        Returns `self.parameters()` unchanged when `no_decay_patterns` is
        empty, so the legacy single-group path is untouched.

        Matching is on a NAME SUBSTRING rather than on module type, because
        the parameters that need exempting are reached by different routes --
        `batch_embedding` is an `nn.Embedding` on the model while
        `conditioning_module.param_generator` is an `nn.Linear` inside the
        encoder -- and a type-based rule would either miss one or sweep in
        every other Linear in the network.

        Returns
        -------
        - Iterable
            Either `self.parameters()` or a list of two param-group dicts.
        """
        if not self.no_decay_patterns:
            return self.parameters()

        decay, no_decay, matched = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(pat in name for pat in self.no_decay_patterns):
                no_decay.append(param)
                matched.append(name)
            else:
                decay.append(param)

        if not no_decay:
            # A pattern that matches nothing is a silent no-op that leaves
            # the very parameters it names being decayed to zero, which is
            # the bug this feature exists to fix. Fail instead.
            raise ValueError(
                f"no_decay_patterns={self.no_decay_patterns} matched no "
                f"parameter. Available names include: "
                f"{[n for n, _ in list(self.named_parameters())[:8]]}..."
            )
        print(
            f"weight decay: {self.weight_decay} on {len(decay)} tensors, "
            f"0.0 on {len(no_decay)} exempt ({', '.join(sorted(matched))})"
        )
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build the bare optimizer.

        Returns
        -------
        - torch.optim.Optimizer
            The configured optimizer.
        """
        # TODO: Add support for multiple optimizers
        if self.optimizer_name == 'adam':
            # Parameter groups. One group unless `no_decay_patterns` is set,
            # which keeps the legacy path byte-identical.
            params = self._param_groups()
            # `fused=True` (when configured) coalesces parameter updates
            # into a single CUDA kernel. Best-effort: if PyTorch refuses
            # the request (CPU params, exotic dtype, older PyTorch
            # version, ...), fall back to the unfused path with a
            # warning rather than crashing training.
            try:
                return torch.optim.Adam(
                    params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    fused=self.fused,
                )
            except (RuntimeError, TypeError, ValueError) as e:
                if self.fused:
                    print(
                        f"WARNING: torch.optim.Adam(fused=True) failed "
                        f"({type(e).__name__}: {e}); falling back to "
                        f"fused=False."
                    )
                return torch.optim.Adam(
                    params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
        else:
            raise NotImplementedError(f'Optimizer {self.optimizer_name} not implemented')


    def common_step(
            self,
            batch_loss_data: dict,
            batch_size: int,
            mode: Literal['train', 'val'] = 'train',
        ) -> torch.Tensor:
        """
        Compute and log the loss for a model for a given batch during training or validation.

        Parameters
        ----------
        - batch_loss_data: dict
            The data required to compute the loss.
        - batch_size: int
            The size of the batch.
        - mode: Literal['train', 'val']
            The mode of the fit process (train, val).

        Returns
        -------
        - torch.Tensor
            The computed loss for the current batch.
        """
        # `mode` MUST be forwarded. `criterion` defaults to mode='train' and
        # logs every per-term loss as f"{mode}_{loss_fn_name}", so omitting it
        # made the validation pass log its per-term losses under `train_*`
        # keys. Those are then overwritten by the real training values before
        # anything reads `callback_metrics` (verified: the per-term train
        # numbers are bit-identical in runs with and without a validation
        # pass), so no reported figure was ever wrong -- but it meant NO
        # per-term `val_*` metric existed at all, only the total `val_loss`.
        # On an 80-epoch run that is the difference between seeing which term
        # drives the generalisation gap and seeing only that there is one.
        loss_value = self.criterion(
            loss_data=batch_loss_data,
            curr_batch_size=batch_size,
            mode=mode,
        )

        self.log(
            name=f'{mode}_loss',
            value=loss_value,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss_value


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the model. The batch of nodes may be the entire set of nodes in the graph or a subset of nodes.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.

        Returns
        -------
        - unnormalized_logits: torch.Tensor
            The unnormalized logits of the model.
        """
        raise NotImplementedError('Forward pass should be implemented by the subclass')


    def _print_epoch_stats(self) -> None:
        """
        Print the epoch stats for the current epoch.
        """
        # print logged values during the current epoch segregated by loss terms and metrics
        logged_value_types = {
            'train': [],
            'val': [],
            'other': [],
            }
        for logged_value_name in self.trainer.callback_metrics.keys():
            if logged_value_name.startswith('train'):
                logged_value_types['train'].append(logged_value_name)
            elif logged_value_name.startswith('val'):
                logged_value_types['val'].append(logged_value_name)
            else:
                logged_value_types['other'].append(logged_value_name)

        print(f"----------------Train-----------------")        
        for loss_term_name in logged_value_types['train']:
                print(f"{loss_term_name}: {self.trainer.callback_metrics[loss_term_name]}")
        print("--------------------------------------------\n")
        
        print(f"----------------Validation-----------------")        
        for loss_term_name in logged_value_types['val']:
                print(f"{loss_term_name}: {self.trainer.callback_metrics[loss_term_name]}")
        print("--------------------------------------------\n")
        
        if len(logged_value_types['other']) > 0:
            print(f"----------------Other-----------------")        
            for loss_term_name in logged_value_types['other']:
                print(f"{loss_term_name}: {self.trainer.callback_metrics[loss_term_name]}")
            print("--------------------------------------------\n")
        
        print(f"--------------------------------End of Epoch {self.current_epoch}--------------------------------------\n\n")
        
        return


    def _update_on_train_epoch_end_logs_df(self) -> None:
        """
        Update the on_train_epoch_end_logs_df dataframe with the logged values for the current epoch.
        """
        logged_values_keys = list(self.trainer.callback_metrics.keys())
        logged_values_values = [value.item() for value in self.trainer.callback_metrics.values()]
        logged_values_dict = dict(zip(logged_values_keys, logged_values_values))
        logged_values_dict['epoch'] = self.current_epoch
        self.on_train_epoch_end_logs_df = pd.concat(
            [
                self.on_train_epoch_end_logs_df,
                pd.DataFrame(
                    [logged_values_dict],
                    columns=list(logged_values_dict.keys())
                )
            ],
            ignore_index=True
        )
        return


    def compute_metrics(
        self,
        mode: Literal['train', 'val', 'test'] = 'val',
    ) -> dict:
        """
        Compute metrics for a given mode. The mode determines the dataloader and inference data cache used to compute the metrics.

        Returns
        -------
        dict
            Metric-name -> value. Returns `{}` when the relevant
            `*_metrics_list` is empty (e.g. losses-only training mode
            where Pearson computation is turned off) — callers can
            iterate the dict unconditionally.

        Parameters
        ----------
        - mode: Literal['train', 'val', 'test']
            The mode of the fit process (train, val, test).
        """
        # 1) Get the dataloader for the given mode
        dataloader = getattr(self.trainer.datamodule, f'{mode}_dataloader')()

        # 2) Get the data and model outputs cached during the steps of the given mode
        inference_data_cache = getattr(self, f'{mode}_inference_data_cache')

        # self.train_metrics_list is a list of metrics specified in the train config file
        # this is used to track the model performance at the end of each epoch
        # currently, the trainer.validate() and trainer.test() methods also use this list
        # a fuller list of metrics must be manually computed separately
        # TODO: Add support for computing this full list
        if mode == 'train' or mode == 'val':
            metrics_list = self.train_metrics_list
        elif mode == 'test':
            metrics_list = self.test_metrics_list

        if len(metrics_list) > 0:
            # Guard: if the dataloader for this mode produced no batches
            # (e.g. val_dataloader is empty because `val_batches=[]` AND
            # `train_val_cell_split=0`), the cache lists are empty and
            # `torch.cat([])` would raise. Return an empty metrics dict
            # so the caller's logging loop is a no-op for this mode.
            if any(len(inference_data_cache[k]) == 0 for k in self.cache_keys):
                # Reset the cache for the next epoch and bail out.
                for key in self.cache_keys:
                    inference_data_cache[key] = []
                return {}

            # 3) Concatenate the inference data cache
            for key in self.cache_keys:
                inference_data_cache[key] = torch.cat(inference_data_cache[key], dim=0)
            inference_data_cache['edge_index'] = dataloader.data.edge_index

            # 4) Convert the inference data to an AnnData object. Pass
            # the per-AnnData `obs` DataFrames (collected at blob-build
            # time and stashed on the datamodule) so every column from
            # every input AnnData ends up on the inference output adata,
            # NaN-filled for cells from sources that don't carry the column.
            adata = inference_data_dict_to_adata(
                inference_data=inference_data_cache,
                label_categories_dict=None,
                obs_per_batch_id=getattr(
                    self.trainer.datamodule, 'obs_per_batch_id', None,
                ),
            )

            # 5) Compute the benchmarking metrics.
            # `estimate_adj_kwargs` is only meaningful for the legacy MLP
            # adjacency-decoder path; the dual model and any other variant
            # without that decoder doesn't carry it. Default to {} so we
            # don't crash here; metrics that genuinely need it will skip
            # themselves on missing inputs.
            metrics_dict = compute_benchmarking_metrics(
                adata=adata,
                metrics=metrics_list,
                **self.loss_kwargs.get('estimate_adj_kwargs', {}),
            )
            
            # 6) Clear the inference data cache
            for key in self.cache_keys:
                inference_data_cache[key] = []

            return metrics_dict

        # `metrics_list` was empty (e.g. losses-only training mode set
        # via the SQUINT_WITH_PEARSON=0 default, where `train_metrics_list`
        # / `test_metrics_list` are both `[]`). Skip the AnnData
        # reconstruction + Pearson loop entirely and return an empty
        # dict so callers can keep iterating unconditionally — earlier
        # the implicit `None` return crashed `on_validation_epoch_end`'s
        # `for key, value in metrics_dict.items():` loop.
        #
        # CRITICAL: even though we're skipping the metrics computation,
        # we MUST still drain the inference cache. `_step` appends to
        # this cache on every training/val step (cell_emb, X_hat, etc. —
        # GPU tensors), and the metrics path was the only consumer that
        # cleared it. Without this clear, GPU memory grows linearly
        # until OOM (~75 s on a ~140 GiB H100 at batch_size=1024). The
        # symptom is a steady upward slope in the wandb GPU memory
        # panel, ending in `torch.cuda.OutOfMemoryError` deep inside
        # the adjacency-BCE loss (the largest per-step allocation).
        for key in self.cache_keys:
            inference_data_cache[key] = []
        return {}


    def on_train_epoch_start(self) -> None:
        """
        Pytorch Lightning hook that is executed at the start of each training epoch.

        Notes
        -----
        - We use this hook to print the start of the current training epoch.
        """        
        print(f"--------------------------------Start of Epoch {self.current_epoch}--------------------------------------")
        self._epoch_t0 = time.time()
        # Anchor the step counter to this epoch as well as the clock. Setting
        # only `_epoch_t0` meant that after the first epoch boundary the elapsed
        # time restarted while `global_step` did not, so the heartbeat divided a
        # cumulative step count by a fresh elapsed and reported nonsense -- the
        # corpus run printed "208 steps/s, elapsed 0.26 h" at step 196,000
        # against a true ~8 steps/s. Cosmetic, but a progress line that lies is
        # worse than none.
        self._epoch_step0 = int(self.global_step)

        # call the parent class method to complete default behavior
        return super().on_train_epoch_start()


    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self._maybe_log_heartbeat(outputs)
        return super().on_train_batch_end(outputs, batch, batch_idx)


    @staticmethod
    def _tensor_census(top: int = 12) -> str:
        """
        What torch tensors are alive, grouped by shape, biggest first.

        Diagnostic only, behind `SQUINT_TENSOR_CENSUS=1`, because it walks the
        whole GC heap. It exists because the corpus run leaks ~9 MB per step
        that survived every elimination -- not the per-block NeighborLoader and
        its workers, not the prefetch thread, not page cache, not the inference
        cache (skipped when `train_metrics_list` is empty), not allocator
        fragmentation (tuning glibc removed only 30%). `tracemalloc` cannot see
        it either: tensor storage is allocated in C++, so Python-level tracing
        reports nothing. Walking the GC heap and summing `element_size() *
        nelement()` per shape does show it, and names the shape -- which is
        usually enough to identify the holder.
        """
        import gc

        import torch as _t
        by_shape: dict = {}
        total = 0
        for o in gc.get_objects():
            try:
                if not isinstance(o, _t.Tensor):
                    continue
                nbytes = o.element_size() * o.nelement()
                key = (tuple(o.shape), str(o.device), str(o.dtype),
                       o.requires_grad, o.grad_fn is not None)
            except Exception:  # noqa: BLE001 - dead/exotic tensors
                continue
            e = by_shape.setdefault(key, [0, 0])
            e[0] += 1
            e[1] += nbytes
            total += nbytes
        rows = sorted(by_shape.items(), key=lambda kv: -kv[1][1])[:top]
        out = [f"    tensor census: {total / 2**30:.2f} GiB live in "
               f"{sum(v[0] for v in by_shape.values()):,} tensors"]
        for (shape, dev, dtype, rg, has_graph), (n, b) in rows:
            out.append(f"      {b / 2**30:7.3f} GiB  x{n:<6} {str(shape):26} "
                       f"{dev:6} {dtype:14} grad={rg} graph={has_graph}")
        return "\n".join(out)


    def _maybe_log_heartbeat(self, outputs) -> None:
        """
        Periodic step heartbeat: step, rate and ETA to the step budget.

        WHY. Nothing else prints between epoch boundaries, and on a corpus a
        single epoch is enormous -- 187,562 steps at batch 512 over the 96M
        training cells -- so the whole 200,000-step budget completes inside
        epoch 0 and the log shows one "Start of Epoch 0" and then nothing for
        hours. The only other progress signals are checkpoints (every 20,000
        steps) and wandb, neither visible in the job log, which makes a healthy
        run indistinguishable from a hung one.

        Off unless `heartbeat_every_n_steps` is set (0 disables), so short runs
        and the test suite stay quiet. Split out of `on_train_batch_end` so it
        is testable without constructing a LightningModule.
        """
        n = int(getattr(self, "heartbeat_every_n_steps", 0) or 0)
        if not n or not self.global_step or self.global_step % n:
            return
        t0 = getattr(self, "_epoch_t0", None)
        budget = getattr(self.trainer, "max_steps", -1) or -1
        msg = f"  step {self.global_step:,}"
        if budget and budget > 0:
            msg += f"/{budget:,} ({100.0 * self.global_step / budget:.1f}%)"
        if t0:
            el = time.time() - t0
            # Rate over THIS epoch, and ETA to the BUDGET rather than the epoch
            # end -- the budget is what actually stops the run.
            done = self.global_step - getattr(self, "_epoch_step0", 0)
            rate = done / max(el, 1e-9)
            msg += f" | {rate:.2f} steps/s | elapsed {el / 3600:.2f} h"
            if budget and budget > 0 and rate > 0:
                msg += f" | ETA {(budget - self.global_step) / rate / 3600:.2f} h"
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        try:
            msg += f" | loss {float(loss):.2f}"
        except Exception:  # noqa: BLE001 - loss shape varies by variant
            pass
        # LR, but only when it is actually scheduled. `LearningRateMonitor`
        # logs to the LOGGER, i.e. wandb -- which is exactly the place the job
        # log cannot see, and the job log is the only artefact available while
        # a farm run is in flight. Without this a schedule that armed but
        # stepped on the wrong grid (per-epoch rather than per-step) is
        # indistinguishable from one that worked. Omitted for a constant LR so
        # existing runs' heartbeats are unchanged.
        if getattr(self, "lr_schedule", "none") != "none":
            try:
                msg += f" | lr {self.optimizers().param_groups[0]['lr']:.3e}"
            except Exception:  # noqa: BLE001 - no optimizer during sanity check
                pass
        print(msg, flush=True)
        if os.environ.get("SQUINT_TENSOR_CENSUS") == "1":
            print(self._tensor_census(), flush=True)


    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Pytorch Lightning hook that is executed for each training batch after the on_train_epoch_start() hook but before the on_validation_epoch_start() hook.

        Parameters
        ----------
        - train_batch: torch_geometric.data.Data
            The training batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed loss.

        Notes
        -----
        - We use this hook to define the training step for the model. This is expected to be implemented by the child class.
        """
        raise NotImplementedError('Training step not implemented')


    def on_validation_epoch_start(self) -> None:
        """
        Pytorch Lightning hook that is executed at the start of each validation epoch. During training, this hook is executed within the on_train_epoch_start() and on_train_epoch_end() hooks, but after the training_step() hook is executed for each training batch.

        Notes
        -----
        - We do not use this hook in the base class. It is included for readability.
        """
        return super().on_validation_epoch_start()


    def validation_step(
            self,
            val_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Pytorch Lightning hook that is executed for each validation batch after the on_validation_epoch_start() hook but before the on_validation_epoch_end() hook.

        Parameters
        ----------
        - val_batch: torch_geometric.data.Data
            The validation batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed validation loss.

        Notes
        -----
        - We use this hook to define the validation step for the model. This is expected to be implemented by the child class.
        """
        raise NotImplementedError('Validation step not implemented')


    def on_validation_epoch_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of each validation epoch. During training, this hook is executed within the on_train_epoch_start() and on_train_epoch_end() hooks, but after the validation_step() hook is executed for each validation batch.

        Notes
        -----
        - We use this hook to compute and log metrics for the nodes in the validation set.
        - This 
        """
        # compute metrics for the nodes in the validation set based on cached data
        metrics_dict = self.compute_metrics(mode='val')
        
        # log the metrics for the nodes in the validation set
        for key, value in metrics_dict.items():
            self.log(
                name=f'val_{key}',
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # call the parent class method to complete default behavior
        return super().on_validation_epoch_end()


    def on_train_epoch_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of each training epoch. During training, this hook is executed after the training_step() hook is executed for each training batch and after the on_validation_epoch_start(), validation_step() for each validation batch, and on_validation_epoch_end() hooks are executed.

        Notes
        -----
        - We use this hook to print the loss terms and metrics for the current epoch.
        - We also use this hook to update the on_train_epoch_end_logs_df dataframe which is used to store the loss terms and metrics for all epochs.
        """
        metrics_dict = self.compute_metrics(mode='train')
        for key, value in metrics_dict.items():
            self.log(
                name=f'train_{key}',
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # print the epoch stats to the console
        self._print_epoch_stats()
        
        # write the logged values (loss terms + metrics) to the epoch metrics dataframe
        self._update_on_train_epoch_end_logs_df()

        # call the parent class method to complete default behavior
        return super().on_train_epoch_end()


    def on_fit_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of the training process.

        Notes
        -----
        - We use this hook to save the on_train_epoch_end_logs_df dataframe to a CSV file.
        """
        # save epoch-wise logged loss terms and metrics to a CSV file
        #
        # `self.logger` is None whenever no logger was attached -- which
        # `train()` does deliberately for `cfg['logging']['enabled'] = False`
        # (`_patch_no_logging`, used by measurement runs where wandb adds only a
        # failure mode). That config option had therefore never actually worked:
        # training ran to completion and then this hook raised
        # `AttributeError: 'NoneType' object has no attribute 'experiment'`,
        # losing the run at the very last step.
        #
        # Fall back to the Trainer's `default_root_dir`, which points at the run
        # directory, so the CSV still lands somewhere sensible. Skip entirely
        # only if there is no directory to be had at all.
        log_dir = None
        if self.logger is not None:
            experiment = getattr(self.logger, 'experiment', None)
            log_dir = getattr(experiment, 'dir', None)
            if log_dir is None:
                log_dir = getattr(self.logger, 'log_dir', None) \
                    or getattr(self.logger, 'save_dir', None)
        if log_dir is None:
            trainer = getattr(self, 'trainer', None)
            log_dir = getattr(trainer, 'default_root_dir', None) if trainer else None
        if log_dir is None:
            print("on_fit_end: no logger and no default_root_dir; skipping the "
                  "on_train_epoch_end_logs CSV.")
            return super().on_fit_end()

        results_dir = Path(log_dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        on_train_epoch_end_logs_fname = results_dir / 'on_train_epoch_end_logs.csv'
        self.on_train_epoch_end_logs_df.to_csv(
            path_or_buf=on_train_epoch_end_logs_fname,
            sep=',',
            index=False
        )

        return super().on_fit_end()
    
    
    def on_test_model_eval(self) -> None:
        """
        Pytorch Lightning hook that is executed before the test steps are called.
        """        
        return super().on_test_model_eval()


    def test_step(
            self,
            test_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Pytorch Lightning hook that is executed for each test batch after the on_test_epoch_start() hook but before the on_test_epoch_end() hook. This hook is outside the training loop.

        Parameters
        ----------
        - test_batch: torch_geometric.data.Data
            The test batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed test accuracy.

        Notes
        -----
        - We use this hook to define the test step for the model. This is expected to be implemented by the child class.
        """
        raise NotImplementedError('Test step not implemented')
    
    def on_test_epoch_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of each test epoch.
        """
        metrics_dict = self.compute_metrics(mode='test')
        print("--------------------------------Test Metrics--------------------------------")
        for key, value in metrics_dict.items():
            print(f"test_{key}: {value}")
            self.log(
                name=f'test_{key}',
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        print("--------------------------------End of Test Metrics--------------------------------")

        return super().on_test_epoch_end()