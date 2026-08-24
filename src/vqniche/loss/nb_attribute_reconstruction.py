"""
Negative binomial attribute reconstruction loss function.
"""

from typing import Literal, Optional

import torch
import torch.nn.functional as F

from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features


def nb_attribute_reconstruction_loss(
        pred_attr: torch.Tensor,
        target_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        dispersion: torch.Tensor,
        k_hop_nb_loss: Literal[0, 1] = 0,
        wt_attr_reconstr: float = 0.1,
        gene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """
    Compute the negative binomial (NB) loss between the estimated attributes from the decoder module and the target attributes.

    Parameters
    ----------
    pred_attr: torch.Tensor
        The output from the attribute decoder module.
        Dimensions: (batch_size, num_genes)
    target_attr: torch.Tensor
        The target attributes.
        Dimensions: (batch_size, num_genes)
    edge_index: torch.Tensor
        The edge index of the graph.
        Dimensions: (2, num_edges)
    batch_size: int
        The size of the batch.
    dispersion: torch.Tensor
        The dispersion parameter.
        Dimensions: (num_genes,)
    gene_mask: Optional[torch.Tensor]
        Cross-panel measured-gene mask, `True` where the cell's section measured
        that gene. Unmeasured entries are zeroed before the per-cell sum, so they
        contribute nothing; without it their (predicted, target=0) pairs would be
        scored as if observed. Must already be sliced to `batch_size`, like
        `pred_attr` / `target_attr`. `None` on the single-panel path leaves the
        reduction untouched.
        Dimensions: (batch_size, num_genes)
    k_hop_nb_loss: Literal[0, 1]
        The number of hops to consider for the neighbor features. 0 indicates individual node attributes, 1 indicates 1-hop neighbor features.
    wt_attr_reconstr: float
        The scaling factor for the node attribute reconstruction loss.

    Returns
    -------
    nb_loss: torch.Tensor
        The computed negative binomial loss.

    Notes
    -----
    - This implementation sets one dispersion parameter per gene.
    - mu (mean of the NB distribution) is the predicted attribute.
    - theta (dispersion parameter) is a learnable parameter.
    - target_attr is the raw count data.
    - NB loss seeks to estimate the true counts conditioned on the predicted attributes.

    References:
    ----------
    - NicheCompass --> https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/modules/losses.py
    - scvi-tools --> https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py#L205
    """
    # if k_hop_nb_loss == 1:
    #     pred_attr = aggregate_1hop_neighbor_features(
    #                     X=pred_attr,
    #                     edge_index=edge_index,
    #                     return_mean=False,
    #                 )
    #     target_attr = aggregate_1hop_neighbor_features(
    #                     X=target_attr,
    #                     edge_index=edge_index,
    #                     return_mean=False,
    #                 )

    # pred_attr = pred_attr[:batch_size]
    # target_attr = target_attr[:batch_size]

    # NOTE: do NOT detach this term — it is the only place where the gradient
    # of the NB log-likelihood w.r.t. the predicted mean (mu) flows through the
    # `dispersion * log(theta + mu)` component. Detaching it (previous code)
    # silently dropped a gradient term and biased the loss. This matches the
    # scvi-tools reference implementation of `log_nb_positive`.
    log_theta_mu_eps = torch.log(dispersion + pred_attr + 1e-8)
    log_likelihood_nb = (
        dispersion * (torch.log(dispersion + 1e-8) - log_theta_mu_eps)
        + target_attr * (torch.log(pred_attr + 1e-8) - log_theta_mu_eps)
        + torch.lgamma(target_attr + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(target_attr + 1))

    # ---- reduction ---------------------------------------------------------
    # Cross-panel: sum over the MEASURED genes only. Zeroing the unmeasured
    # terms before the sum keeps this the log-likelihood of the OBSERVED data --
    # a cell whose section measures 169 genes contributes less than one
    # measuring 4,949 because it carries less information, which is correct
    # rather than an imbalance to correct for.
    #
    # A MEAN over measured genes was the earlier plan and is wrong here: it
    # would divide the term by ~W. R0 logs nb_cell ~= 844 against an adjacency
    # BCE of ~517, so a mean over 4,948 genes gives ~0.17 and the BCE would
    # outweigh reconstruction by ~3,000x -- training would stop reconstructing
    # unless every wt_* were retuned. Panel imbalance, if it ever proves
    # harmful, belongs in block composition or an explicit per-panel weight, not
    # in a distorted likelihood.
    #
    # With an all-ones mask this is arithmetically identical to the unmasked
    # expression, which is what keeps the single-panel path bit-identical.
    if gene_mask is not None:
        if gene_mask.shape != log_likelihood_nb.shape:
            raise ValueError(
                f"gene_mask {tuple(gene_mask.shape)} must match the NB term "
                f"{tuple(log_likelihood_nb.shape)}. A mismatch would score the "
                f"wrong genes -- most likely the mask was not sliced to "
                f"`batch_size` the way pred_attr / target_attr are."
            )
        log_likelihood_nb = log_likelihood_nb * gene_mask

    nb_loss = torch.mean(-log_likelihood_nb.sum(-1))

    return nb_loss * wt_attr_reconstr


def nb_nbr_attribute_reconstruction_loss(
        pred_attr_nbr: torch.Tensor,
        target_attr_nbr: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        dispersion: torch.Tensor,
        k_hop_nb_loss: int = 0,
        wt_attr_reconstr: float = 0.1,
        gene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """
    Thin wrapper around `nb_attribute_reconstruction_loss` for the
    neighbourhood branch when `recon_mode='both'`.

    The loss dispatcher extracts tensors from the step's loss_data dict by
    matching keyword names to dict keys.  The neighbourhood targets are stored
    under `pred_attr_nbr` / `target_attr_nbr` to avoid colliding with the
    per-cell keys (`pred_attr` / `target_attr`). This wrapper accepts those
    suffixed names and forwards them to the canonical loss under the expected
    positional names.

    The upstream code (training_step / validation_step) already computes the
    1-hop aggregation before storing the pair, so `k_hop_nb_loss` is always
    passed as 0 by the dispatcher — the aggregation is NOT repeated here.
    """
    return nb_attribute_reconstruction_loss(
        pred_attr=pred_attr_nbr,
        target_attr=target_attr_nbr,
        edge_index=edge_index,
        batch_size=batch_size,
        dispersion=dispersion,
        k_hop_nb_loss=k_hop_nb_loss,
        wt_attr_reconstr=wt_attr_reconstr,
        # Same mask as the cell branch: spatial graphs never cross sections, so
        # a cell's 1-hop neighbours always share its panel and the neighbourhood
        # mean is taken over the identical measured-gene set.
        gene_mask=gene_mask,
    )


def nb_nbr_attribute_reconstruction_loss_dual(
        pred_attr_nbr: torch.Tensor,
        target_attr_nbr: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        dispersion_niche: torch.Tensor,
        k_hop_nb_loss: int = 0,
        wt_attr_reconstr: float = 0.1,
        gene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """
    Same as `nb_nbr_attribute_reconstruction_loss` but reads its NB-
    dispersion parameter from a *separate* tensor named `dispersion_niche`,
    so the niche branch in VQNiche_Dual can have its own learnable
    per-gene dispersion that is decoupled from the cell-branch dispersion.

    Why a separate dispersion is essential for the dual model
    ---------------------------------------------------------
    The two NB likelihoods in VQNiche_Dual see targets with very
    different variance structure:

      - Cell-branch target  = per-cell raw counts
                              (high variance, lots of zeros)
                              -> optimal theta is SMALL (high overdispersion)
      - Niche-branch target = 1-hop neighbourhood-mean of raw counts
                              (smoother by ~8-9x via averaging)
                              -> optimal theta is LARGE (low overdispersion)

    With a *shared* dispersion the niche branch's gradient on theta is
    cleaner (smooth target, easier to fit) so it dominates: theta drifts
    up over training and the cell-branch NB likelihood degrades even
    when the cell decoder predictions are static. Decoupling the two
    dispersion parameters removes this gradient conflict entirely.
    """
    return nb_attribute_reconstruction_loss(
        pred_attr=pred_attr_nbr,
        target_attr=target_attr_nbr,
        edge_index=edge_index,
        batch_size=batch_size,
        dispersion=dispersion_niche,
        k_hop_nb_loss=k_hop_nb_loss,
        wt_attr_reconstr=wt_attr_reconstr,
        # Same mask as the cell branch: spatial graphs never cross sections, so
        # a cell's 1-hop neighbours always share its panel and the neighbourhood
        # mean is taken over the identical measured-gene set.
        gene_mask=gene_mask,
    )