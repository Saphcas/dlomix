import logging

import numpy as np
import tensorflow as tf
import torch

from dlomix.losses.intensity import (
    masked_pearson_correlation_distance,
    masked_spectral_distance,
)
from dlomix.losses.intensity_torch import (
    masked_pearson_correlation_distance as masked_pearson_correlation_distance_torch,
)
from dlomix.losses.intensity_torch import (
    gaussian_nll,
    masked_spectral_distance as masked_spectral_distance_torch,
)

logger = logging.getLogger(__name__)


# ------------------ intensity - masked spectral distance ------------------


def test_tf_torch_equivalence_masked_spectral_distance():
    y_true = [[0.1, 0.2, 0.3]]
    y_pred = [list(reversed(y_true[0]))]

    sa_tf = masked_spectral_distance(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    )
    sa_torch = masked_spectral_distance_torch(
        torch.tensor(y_true), torch.tensor(y_pred)
    )

    logger.info(
        f"Spectral Angle: for tf: {sa_tf.numpy()} vs for torch: {sa_torch.numpy()}"
    )

    assert np.allclose(
        sa_tf.numpy(), sa_torch.numpy()
    )  # alternatively try np.array_equiv(A,B)


# ------------------ intensity - masked pearson correlation distance ------------------


def test_tf_torch_equivalence_masked_pearson_correlation_distance():
    y_true = [[0.1, 0.2, 0.3]]
    y_pred = [list(reversed(y_true[0]))]

    pc_tf = masked_pearson_correlation_distance(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    )
    pc_torch = masked_pearson_correlation_distance_torch(
        torch.tensor(y_true), torch.tensor(y_pred)
    )

    logger.info(
        f"Masked Pearson Correlation Distance: for tf: {pc_tf.numpy()} vs for torch: {pc_torch.numpy()}"
    )

    assert np.allclose(
        pc_tf.numpy(), pc_torch.numpy()
    )  # alternatively try np.array_equiv(A,B)


# add tests for IonMobLoss

# ------------------ intensity - zero-inflated Gaussian NLL ------------------


def test_gaussian_nll_masks_impossible_fragments_and_uses_log_targets():
    eps = 1e-7
    y_true = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.5, 0.7, -1.0],
            [0.0, 2.0, 1.0, -1.0, -1.0, -1.0],
        ]
    )
    encoded_sequence = torch.tensor(
        [
            [21, 1, 2, 3, 22, 0],
            [21, 1, 2, 22, 0, 0],
        ]
    )

    safe_target = torch.where(y_true > 0, y_true, torch.ones_like(y_true))
    y_mean_pred = torch.log(safe_target + eps)
    y_log_var_pred = torch.zeros_like(y_true)
    presence_logits = torch.zeros_like(y_true)

    loss = gaussian_nll(
        y_true,
        y_mean_pred,
        y_log_var_pred,
        presence_logits,
        encoded_sequence,
        fragments_per_cleavage=2,
    )

    # The default averages each component within each peptide. With zero
    # residuals and zero logits, both peptides have the same component means.
    expected = torch.log(torch.tensor(2.0)) + 0.5 * torch.log(
        torch.tensor(2.0 * np.pi)
    )

    assert torch.allclose(loss, expected, atol=1e-6)


def test_gaussian_nll_handles_batches_without_present_fragments():
    y_true = torch.tensor([[0.0, 0.0, -1.0, -1.0]])
    encoded_sequence = torch.tensor([[21, 1, 2, 22]])
    y_mean_pred = torch.zeros_like(y_true)
    y_log_var_pred = torch.zeros_like(y_true)
    presence_logits = torch.zeros_like(y_true)

    loss = gaussian_nll(
        y_true,
        y_mean_pred,
        y_log_var_pred,
        presence_logits,
        encoded_sequence,
        fragments_per_cleavage=2,
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(0.0), torch.tensor(0.0), reduction="sum"
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gaussian_nll_raises_when_no_valid_fragments_remain():
    y_true = torch.tensor([[-1.0, -1.0]])
    encoded_sequence = torch.tensor([[21, 1, 2, 22]])
    y_mean_pred = torch.zeros_like(y_true)
    y_log_var_pred = torch.zeros_like(y_true)
    presence_logits = torch.zeros_like(y_true)

    try:
        gaussian_nll(
            y_true,
            y_mean_pred,
            y_log_var_pred,
            presence_logits,
            encoded_sequence,
            fragments_per_cleavage=2,
        )
    except ValueError as exc:
        assert "No valid fragment ions remain" in str(exc)
    else:
        raise AssertionError("Expected gaussian_nll to raise ValueError")



def test_gaussian_nll_component_mean_normalizes_components_separately():
    y_true = torch.tensor([[1.0], [0.0]])
    encoded_sequence = torch.tensor([[21, 1, 2, 22], [21, 1, 2, 22]])
    y_mean_pred = torch.zeros_like(y_true)
    y_log_var_pred = torch.zeros_like(y_true)
    presence_logits = torch.zeros_like(y_true)

    loss = gaussian_nll(
        y_true,
        y_mean_pred,
        y_log_var_pred,
        presence_logits,
        encoded_sequence,
        fragments_per_cleavage=1,
        normalization="component_mean",
    )

    expected = torch.log(torch.tensor(2.0)) + 0.5 * torch.log(
        torch.tensor(2.0 * np.pi)
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gaussian_nll_softplus_variance_is_finite_and_floored():
    y_true = torch.tensor([[1.0]])
    encoded_sequence = torch.tensor([[21, 1, 2, 22]])
    y_mean_pred = torch.zeros_like(y_true)
    raw_variance = torch.tensor([[-100.0]])
    presence_logits = torch.zeros_like(y_true)
    min_variance = 0.25

    loss = gaussian_nll(
        y_true,
        y_mean_pred,
        raw_variance,
        presence_logits,
        encoded_sequence,
        fragments_per_cleavage=1,
        variance_parameterization="softplus_variance",
        min_variance=min_variance,
    )

    expected_gaussian = 0.5 * torch.log(torch.tensor(2.0 * np.pi * min_variance))
    expected = torch.log(torch.tensor(2.0)) + expected_gaussian
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected, atol=1e-5)



def test_gaussian_nll_per_peptide_component_mean_normalizes_positive_ions_separately():
    y_true = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    encoded_sequence = torch.tensor([[21, 1, 2, 22], [21, 1, 2, 22]])
    y_mean_pred = torch.zeros_like(y_true)
    y_log_var_pred = torch.zeros_like(y_true)
    presence_logits = torch.zeros_like(y_true)

    loss = gaussian_nll(
        y_true,
        y_mean_pred,
        y_log_var_pred,
        presence_logits,
        encoded_sequence,
        fragments_per_cleavage=2,
        normalization="per_peptide_component_mean",
    )

    expected = torch.log(torch.tensor(2.0)) + 0.5 * torch.log(
        torch.tensor(2.0 * np.pi)
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gaussian_nll_per_peptide_component_mean_excludes_all_zero_peptides():
    y_true = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    encoded_sequence = torch.tensor([[21, 1, 2, 22], [21, 1, 2, 22]])
    y_mean_pred = torch.zeros_like(y_true)
    y_log_var_pred = torch.zeros_like(y_true)
    presence_logits = torch.zeros_like(y_true)

    loss = gaussian_nll(
        y_true,
        y_mean_pred,
        y_log_var_pred,
        presence_logits,
        encoded_sequence,
        fragments_per_cleavage=2,
        normalization="per_peptide_component_mean",
    )

    # The Gaussian component is averaged only over the peptide with positives.
    expected = torch.log(torch.tensor(2.0)) + 0.5 * torch.log(
        torch.tensor(2.0 * np.pi)
    )
    assert torch.allclose(loss, expected, atol=1e-6)
