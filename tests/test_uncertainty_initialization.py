from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from dlomix.uncertainty_initialization import (
    LOG_INTENSITY_EPSILON,
    StreamingChannelStatistics,
    apply_empirical_head_initialization,
)


class _UncertaintyHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_mean_regressor = nn.Sequential(
            OrderedDict([("output_dense", nn.Linear(2, 6))])
        )
        self.log_var_regressor = nn.Sequential(
            OrderedDict([("output_dense", nn.Linear(2, 6))])
        )
        self.presence_regressor = nn.Sequential(
            OrderedDict([("output_dense", nn.Linear(2, 6))])
        )


def test_streaming_channel_statistics_respects_masks_and_flattened_channels():
    statistics = StreamingChannelStatistics(channels=2, epsilon=LOG_INTENSITY_EPSILON)
    statistics.update(np.array([[1.0, 0.0, -1.0, 2.0]]))
    statistics.update(np.array([[0.5, 1.0, 0.0, -1.0]]))

    output = statistics.to_dict()
    assert output["channels"][0]["valid_count"] == 3
    assert output["channels"][0]["present_count"] == 2
    assert output["channels"][0]["presence_probability"] == 2.0 / 3.0
    assert output["channels"][1]["valid_count"] == 3
    assert output["channels"][1]["present_count"] == 2
    assert output["channels"][1]["presence_probability"] == 2.0 / 3.0

    expected_channel_zero = np.log(np.array([1.0, 0.5]) + LOG_INTENSITY_EPSILON)
    expected_channel_one = np.log(np.array([2.0, 1.0]) + LOG_INTENSITY_EPSILON)
    assert np.isclose(output["mean_log_intensity"][0], expected_channel_zero.mean())
    assert np.isclose(output["mean_log_intensity"][1], expected_channel_one.mean())
    assert np.isclose(
        output["marginal_log_intensity_variance"][0], expected_channel_zero.var()
    )
    assert np.isclose(
        output["marginal_log_intensity_variance"][1], expected_channel_one.var()
    )


def test_empirical_head_initialization_sets_biases_only(tmp_path):
    stats_path = tmp_path / "head_stats.json"
    stats = {
        "mean_log_intensity": [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5],
        "presence_probability": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "marginal_log_intensity_variance": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    model = _UncertaintyHeadModel()
    original_mean_weight = model.log_mean_regressor.output_dense.weight.detach().clone()

    applied = apply_empirical_head_initialization(
        model,
        mean_mode="empirical",
        presence_mode="empirical",
        variance_mode="empirical_marginal",
        stats_path=stats_path,
        variance_parameterization="log_var",
    )

    assert torch.allclose(
        model.log_mean_regressor.output_dense.bias,
        torch.tensor(stats["mean_log_intensity"]),
    )
    probabilities = torch.tensor(stats["presence_probability"])
    assert torch.allclose(
        model.presence_regressor.output_dense.bias,
        torch.logit(probabilities),
    )
    assert torch.allclose(
        model.log_var_regressor.output_dense.bias,
        torch.log(torch.tensor(stats["marginal_log_intensity_variance"])),
    )
    assert torch.equal(model.log_mean_regressor.output_dense.weight, original_mean_weight)
    assert applied["marginal_target_variance"] == stats["marginal_log_intensity_variance"]


def test_softplus_variance_initialization_maps_back_to_target_variance(tmp_path):
    stats_path = tmp_path / "head_stats.json"
    target_variance = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    stats_path.write_text(
        json.dumps(
            {
                "mean_log_intensity": [-1.0] * 6,
                "presence_probability": [0.5] * 6,
                "marginal_log_intensity_variance": target_variance,
            }
        ),
        encoding="utf-8",
    )
    model = _UncertaintyHeadModel()
    min_variance = 0.1

    apply_empirical_head_initialization(
        model,
        variance_mode="empirical_marginal",
        stats_path=stats_path,
        variance_parameterization="softplus_variance",
        min_variance=min_variance,
    )

    recovered = min_variance + torch.nn.functional.softplus(
        model.log_var_regressor.output_dense.bias
    )
    assert torch.allclose(recovered, torch.tensor(target_variance), atol=1e-6)
