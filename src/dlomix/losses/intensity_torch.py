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
    
    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = 1e-7

    # in previous masking y_true = 0 will be 0, and y_true = -1 will be 0. 
    # unlike previous methods we do not want to make y_missingness = 0 based on y_true,
    # mask with ((y_pred + 1) * y_pred) / (y_pred + 1 + epsilon)??
    # Or y_pred / (y_pred + 1 + epsilon)??
    # try new masking

    # Masking
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon) # for presence
    y_true[y_true==-1] = 0 # to allow log()
    true_log_masked = torch.log(y_true + epsilon) # see if logic is sound
    mean_log_masked = torch.log(y_mean_pred + epsilon)
    var_log_masked = torch.log(torch.pow(y_var_pred, 2) + epsilon)
    missingness_masked = y_missingness_pred + epsilon # Might need to contain between 0 to 1

    before_sum = (torch.add(torch.mul(torch.exp(-var_log_masked), torch.pow(torch.sub(true_log_masked, mean_log_masked), 2)), var_log_masked))
    #test1 = torch.exp(-var_log_masked)
    #test2 = torch.sub(true_log_masked, mean_log_masked)
    #test3 = torch.pow(test2, 2)
    #test4 = torch.mul(test1, test3)
    #test5 = torch.add(test4, var_log_masked)
    nll_loss = torch.sum(before_sum, 1) * (1/2) # Each batch elements loss
    total_nll_loss = torch.sum(nll_loss) # Sum of each batch element loss

    y_zero_presence = torch.sub(1, missingness_masked)
    y_larger_presence = torch.mul(missingness_masked, torch.normal(mean_log_masked, torch.abs(var_log_masked)))
    # Try torch.where for the condition y==0, currently y_zero starts positive and y_larger can contain negative values
    # Presumably due to log mean being negative
    presence_pred = torch.abs(torch.where(y_true == 0, y_zero_presence, y_larger_presence)) # Should probably find better solution for negative presence predictions
    presence_loss = - torch.sum(torch.log(presence_pred))
    total_presence_loss = torch.sum(presence_loss)

    total_loss = total_nll_loss + total_presence_loss

    return total_loss
