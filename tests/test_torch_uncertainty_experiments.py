"""Regression tests for optional uncertainty-aware optimization modes."""

from __future__ import annotations

import copy
import os

os.environ.setdefault("DLOMIX_BACKEND", "torch")

import torch

from dlomix.losses.intensity_torch import gaussian_nll
from dlomix.models import PrositIntensityUncertaintyPredictor


def _loss_inputs():
    target = torch.tensor([[0.5, 0.0, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]])
    sequence = torch.tensor([[21, 1, 2, 3, 22], [21, 4, 5, 6, 22]])
    mean = torch.tensor(
        [[-0.4, 0.2, -1.0, 0.3], [-0.7, -0.5, 0.4, 0.1]],
        requires_grad=True,
    )
    log_variance = torch.tensor(
        [[0.2, -0.1, 0.4, 0.3], [0.5, -0.3, 0.1, 0.2]],
        requires_grad=True,
    )
    logits = torch.zeros_like(mean, requires_grad=True)
    return target, sequence, mean, log_variance, logits


def _model(mode: str) -> PrositIntensityUncertaintyPredictor:
    return PrositIntensityUncertaintyPredictor(
        seq_length=5,
        len_fion=2,
        embedding_output_dim=2,
        recurrent_layers_sizes=(2, 4),
        regressor_layer_size=4,
        dropout_rate=0.0,
        latent_dropout_rate=0.0,
        input_keys={"SEQUENCE_KEY": "sequence"},
        meta_data_keys=None,
        with_termini=True,
        faithful_mode=mode,
    )


def _flatten_grads(module: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.reshape(-1)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(values) if values else torch.empty(0)


def test_beta_zero_is_exact_baseline():
    target, sequence, mean, log_variance, logits = _loss_inputs()
    baseline = gaussian_nll(
        target, mean, log_variance, logits, sequence, fragments_per_cleavage=2
    )
    beta_zero = gaussian_nll(
        target,
        mean,
        log_variance,
        logits,
        sequence,
        fragments_per_cleavage=2,
        beta_nll_beta=0.0,
    )
    assert torch.equal(baseline, beta_zero)


def test_beta_weight_uses_detached_variance():
    target, sequence, mean, log_variance, logits = _loss_inputs()
    beta = 0.5
    actual = gaussian_nll(
        target,
        mean,
        log_variance,
        logits,
        sequence,
        fragments_per_cleavage=2,
        presence_weight=0.0,
        beta_nll_beta=beta,
    )
    actual.backward()
    actual_log_variance_grad = log_variance.grad.detach().clone()

    target, sequence, mean, log_variance, logits = _loss_inputs()
    present = target > 0
    residual_squared = (torch.log(target[present] + 1e-7) - mean[present]).square()
    standard_nll = 0.5 * (
        torch.exp(-log_variance[present]) * residual_squared
        + log_variance[present]
        + torch.log(torch.tensor(2.0 * torch.pi))
    )
    expected = (
        torch.exp(log_variance[present].detach()).pow(beta) * standard_nll
    ).mean()
    expected.backward()
    assert torch.allclose(actual_log_variance_grad[present], log_variance.grad[present])


def test_beta_weighting_happens_before_each_normalization():
    target, sequence, mean, log_variance, logits = _loss_inputs()
    beta = 1.0
    present = target > 0
    valid = target >= 0
    per_ion = torch.zeros_like(target)
    residual_squared = (torch.log(target[present] + 1e-7) - mean[present]).square()
    nll = 0.5 * (
        torch.exp(-log_variance[present]) * residual_squared
        + log_variance[present]
        + torch.log(torch.tensor(2.0 * torch.pi))
    )
    per_ion[present] = torch.exp(log_variance[present].detach()) * nll
    valid_count = valid.sum(dim=1)
    present_count = present.sum(dim=1)
    expected_by_normalization = {
        "component_mean": per_ion[present].mean(),
        "per_peptide": (per_ion.sum(dim=1) / valid_count).mean(),
        "per_peptide_component_mean": (
            per_ion.sum(dim=1)[present_count > 0]
            / present_count[present_count > 0]
        ).mean(),
    }
    for normalization, expected in expected_by_normalization.items():
        loss = gaussian_nll(
            target,
            mean,
            log_variance,
            logits,
            sequence,
            fragments_per_cleavage=2,
            normalization=normalization,
            presence_weight=0.0,
            beta_nll_beta=beta,
        )
        assert torch.allclose(loss, expected)


def test_variance_prior_matches_each_normalization():
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    sequence = torch.tensor([[21, 1, 2, 3, 22], [21, 4, 5, 6, 22]])
    mean = torch.zeros_like(target)
    logits = torch.zeros_like(target)
    log_variance = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.0, 0.0]])
    common = dict(
        fragments_per_cleavage=2,
        presence_weight=0.0,
        intensity_weight=0.0,
        log_variance_prior_centers=[0.0, 0.0],
        log_variance_prior_weight=1.0,
    )
    component = gaussian_nll(
        target, mean, log_variance, logits, sequence, normalization="component_mean", **common
    )
    per_peptide = gaussian_nll(
        target, mean, log_variance, logits, sequence, normalization="per_peptide", **common
    )
    per_peptide_component = gaussian_nll(
        target,
        mean,
        log_variance,
        logits,
        sequence,
        normalization="per_peptide_component_mean",
        **common,
    )
    assert torch.allclose(component, torch.tensor(30.0 / 4.0))
    assert torch.allclose(per_peptide, torch.tensor((1.0 / 4.0 + 29.0 / 4.0) / 2.0))
    assert torch.allclose(per_peptide_component, torch.tensor((1.0 + 29.0 / 3.0) / 2.0))


def test_prior_channel_count_follows_fragments_per_cleavage():
    target, sequence, mean, log_variance, logits = _loss_inputs()
    try:
        gaussian_nll(
            target,
            mean,
            log_variance,
            logits,
            sequence,
            fragments_per_cleavage=2,
            log_variance_prior_centers=[0.0] * 6,
            log_variance_prior_weight=1.0,
        )
    except ValueError as exc:
        assert "fragment channel (2)" in str(exc)
    else:
        raise AssertionError("Expected mismatched prior centers to fail.")


def test_faithful_variance_head_isolated_from_decoder_and_mean_is_sse():
    torch.manual_seed(7)
    inputs = {"sequence": torch.tensor([[21, 1, 2, 3, 22], [21, 4, 5, 6, 22]])}
    target = torch.tensor([[0.5, 0.0, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]])

    faithful = _model("faithful_variance_only")
    faithful(inputs)  # materialize lazy layers before copying.
    reference = copy.deepcopy(faithful)
    reference.faithful_mode = "off"

    mean, log_variance, logits = faithful(inputs)
    faithful_loss = gaussian_nll(
        target,
        mean,
        log_variance,
        logits,
        inputs["sequence"],
        fragments_per_cleavage=2,
        faithful_mode="faithful_variance_only",
        presence_weight=0.0,
    )
    faithful_loss.backward()
    faithful_decoder_grad = _flatten_grads(faithful.decoder).clone()
    assert torch.count_nonzero(_flatten_grads(faithful.log_var_regressor)) > 0

    reference_mean, _, _ = reference(inputs)
    present = target > 0
    sse_per_ion = torch.zeros_like(target)
    sse_per_ion[present] = 0.5 * (
        torch.log(target[present] + 1e-7) - reference_mean[present]
    ).square()
    present_count = present.sum(dim=1)
    sse_loss = (
        sse_per_ion.sum(dim=1)[present_count > 0]
        / present_count[present_count > 0]
    ).mean()
    sse_loss.backward()
    assert torch.allclose(
        faithful_decoder_grad, _flatten_grads(reference.decoder), atol=1e-6
    )


def test_presence_reaches_decoder_only_in_variance_only_mode():
    torch.manual_seed(11)
    inputs = {"sequence": torch.tensor([[21, 1, 2, 3, 22], [21, 4, 5, 6, 22]])}
    target = torch.tensor([[0.5, 0.0, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]])

    variance_only = _model("faithful_variance_only")
    variance_only(inputs)
    strict = copy.deepcopy(variance_only)
    strict.faithful_mode = "faithful_strict"

    for model, mode in (
        (variance_only, "faithful_variance_only"),
        (strict, "faithful_strict"),
    ):
        mean, log_variance, logits = model(inputs)
        loss = gaussian_nll(
            target,
            mean,
            log_variance,
            logits,
            inputs["sequence"],
            fragments_per_cleavage=2,
            faithful_mode=mode,
            presence_weight=1.0,
            intensity_weight=0.0,
        )
        loss.backward()

    assert torch.count_nonzero(_flatten_grads(variance_only.decoder)) > 0
    assert torch.count_nonzero(_flatten_grads(strict.decoder)) == 0
    assert torch.count_nonzero(_flatten_grads(strict.presence_regressor)) > 0
