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
    y_true: torch.Tensor, y_mean_pred: torch.Tensor, y_var_pred: torch.Tensor, y_missingness_pred: torch.Tensor,
    encoded_sequence: torch.Tensor,
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
        A tensor containing the predicted missingness logits, with the same shape as `y_true`.
    encoded_sequence : torch.Tensor
        Tensor containing the number encoded sequence. N-terminal encoded as 21, and C-terminal
        encoded as 22. Shape is equal to `(batch_size, max_seq_len)`.

    Returns
    -------
    torch.Tensor
        A tensor containing the sum of the GaussianNLL and BCE loss.

    """
    # To avoid numerical instability concerning variance calculation in particular
    epsilon = 1e-7

    # Use encoded sequence to determine sequence length and which tensor values are impossible.
    # Encoded sequence contains the N- and C- terminal as well, they are not included in the intensity predictions.
    # ----- WARNING -----
    # If the encoding numbers for the terminals change the following code will need modification,
    # currently (as seen in PTMS_ALPHABET in constants.py) the N-terminal is encoded as 21, and C-terminal as 22.

    # Collect the length of valid sequences in batch    
    
    # Removes padding, and terminals from the sequences
    encoded_sequence = [tensor[tensor!=0.] for tensor in encoded_sequence]
    encoded_sequence = [tensor[tensor!=21.] for tensor in encoded_sequence]
    encoded_sequence = [tensor[tensor!=22.] for tensor in encoded_sequence]
    # Index in instensity vector calculation
    ind = torch.tensor([(3*2*(len(tensor)-1)-1) for tensor in encoded_sequence])

    # Create a mask to sort which values to send to GaussianNLLLoss()
    mask = torch.zeros_like(y_true)
    mask[(torch.arange(y_true.shape[0]), ind)] = 1
    mask = 1 - mask.cumsum(dim=1)
    valid = mask.bool()

    # Sanity check to hopefully not raise an assert error
    if valid.shape == y_true.shape:
        valid_y_true = y_true[valid]
        valid_y_mean_pred = y_mean_pred[valid]
        valid_y_var_pred = y_var_pred[valid]
    else: 
        print(f"Shape mismatch. valid shape: {valid.shape()}, y shape: {y_true.shape()}")
        print("Using entire vector")
        valid_y_true = y_true
        valid_y_mean_pred = y_mean_pred
        valid_y_var_pred = y_var_pred

    # Missing ions within the sequence will be treated as missing signals (0)
    existing = valid_y_true > 0
    valid_y_true[~existing] = 0

    # To predict missingness; all elements with y_true > 0 will be 0, and 
    # others will be 1 as they are missing.
    # This tensor will only contains 1:s and 0:s afterward

    # Getting bool of which y values are NOT 0 or -1
    present = y_true > 0

    missingness_target = y_true.clone()
    missingness_target[present] = 0
    missingness_target[~present] = 1

    # Clamp variance before GaussianNLL to epsilon as to avoid explosion
    var_pred_masked = valid_y_var_pred + epsilon
    
    # First part of the loss function
    # The tensors will be trained to be means and variances,
    # even if the function is preformed in log space.
    nll = torch.nn.GaussianNLLLoss(eps=epsilon, reduction='mean')
    nll_loss = nll(input=valid_y_mean_pred, target=valid_y_true, var=var_pred_masked)

    #TODO: casting (?) requires BCEWithLogitsLoss, therefore change code back to using logits (done),
    # and add a conversion for the output instead. See WandB logs.
    # Second part of the loss function
    presence_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    presence = presence_loss(y_missingness_pred, missingness_target) 

    total_loss = nll_loss + presence

    return total_loss
