import numpy as np
import torch
import torch.nn.functional as F


def masked_spectral_distance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the masked spectral distance between true and predicted intensity vectors.
    The masked spectral distance is a metric for comparing the similarity between two intensity vectors.

    Masked, normalized spectral angles between true and pred vectors

    > arccos(1*1 + 0*0) = 0 -> SL = 0 -> high correlation

    > arccos(0*1 + 1*0) = pi/2 -> SL = 1 -> low correlation

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : torch.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the masked spectral distance between `y_true` and `y_pred`.

    """

    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = 1e-7

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 norm
    # along last axis / dimension of the tensor
    true_norm = F.normalize(true_masked, p=2, dim=-1)
    pred_norm = F.normalize(pred_masked, p=2, dim=-1)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = (pred_norm * true_norm).sum(dim=-1)
    product = torch.clamp(product, -1.0 + epsilon, 1.0 - epsilon)
    arccos = torch.arccos(product)
    batch_losses = 2 * arccos / np.pi

    return batch_losses.mean()


def masked_pearson_correlation_distance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the masked Pearson correlation distance between true and predicted intensity vectors.
    The masked Pearson correlation distance is a metric for comparing the similarity between two intensity vectors,
    taking into account only the non-negative values in the true values tensor (which represent valid peaks).

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : torch.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the masked Pearson correlation distance between `y_true` and `y_pred`.

    """

    epsilon = 1e-7

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    mx = true_masked.mean()
    my = pred_masked.mean()
    xm, ym = true_masked - mx, pred_masked - my
    r_num = (xm * ym).mean()
    r_den = xm.std(unbiased=False) * ym.std(unbiased=False)

    return 1 - (r_num / r_den)


def gaussian_nll(
    y_true: torch.Tensor, y_mean_pred: torch.Tensor, y_var_pred: torch.Tensor, y_missingness_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calcuates a combined loss of negative log likelihood, and binary cross entropy with logits.
    The NLL loss uses the true vector together with the predicted mean, and variance.
    For the BCE loss the true vector is turned into a categorical vector 
    with 0:s for missing or defined missing (-1) intensities, and 1:s for all intensities > 0.
    The modified true vector is the target for the missingness prediction within the
    BCEWithLogitsLoss() function.

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_mean_pred : torch.Tensor
        A tensor containing the predicted mean values, with the same shape as `y_true`.
    y_var_pred : torch.Tensor
        A tensor containing the predicted variance values, with the same shape as `y_true`.
    y_missingness_pred : torch.Tensor
        A tensor containing the predicted missingness probabilities, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the sum of the GaussianNLL and BCE loss.

    """
    
    # To avoid numerical instability during training on GPUs,
    # epsilon is utilized
    epsilon = 1e-7

    # Getting indicies of which y values are NOT 0 or -1
    present = y_true > 0

    # Masking
    # Cloning y_true to ensure that the in-place changes does not effect other
    # parts of the y_true vector use
    y_true_masked = y_true.clone()
    # Will set -1 values to 0
    y_true_masked[~present] = 0
    missingness_target = torch.clone(y_true_masked)

    # To predict missingness; all elements with y_true > 0 will be 0, and 
    # others will be 1 as they are missing.
    # This tensor will only contains 1:s and 0:s afterward
    missingness_target[present] = 0
    missingness_target[~present] = 1

    # Clamp inputs for GaussianNLL to epsilon as to avoid logarithms of small numbers and 0
    true_clamped = torch.clamp(y_true_masked, min = epsilon)
    mean_clamped = torch.clamp(y_mean_pred, min = epsilon)
    var_clamped = torch.clamp(y_var_pred, min = epsilon)

    # First part of the loss function
    # The tensors will be trained to be means and variances,
    # even if the function is preformed in log space.
    nll = torch.nn.GaussianNLLLoss(eps=epsilon, reduction='mean')
    nll_loss = nll(input=mean_clamped, target=true_clamped, var=var_clamped)

    #TODO: casting (?) requires BCEWithLogitsLoss, therefore change code back to using logits (done),
    # and add a conversion for the output instead. See WandB logs.
    # Second part of the loss function
    presence_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    presence = presence_loss(y_missingness_pred, missingness_target) 

    total_loss = nll_loss + presence

    return total_loss
