"""Implement the single-output PGTuner QPP model used in the paper."""

from __future__ import annotations

import random
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn


class DirectPredictMLP(nn.Module):
    """Build the PGTuner direct-prediction multilayer perceptron."""

    def __init__(self, layer_sizes: list[int]) -> None:
        """Initialize the fixed hidden-layer architecture and weights."""
        super().__init__()
        sizes = [int(value) for value in layer_sizes]
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.BatchNorm1d(sizes[index + 1]))
                layers.append(nn.ReLU(inplace=False))
        layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return the scaled search-cost prediction."""
        return self.net(values)


def seed_everything(seed: int, threads: int) -> None:
    """Set the random seeds and PyTorch thread counts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def transform_features(frame, features: list[str]) -> np.ndarray:
    """Apply the PGTuner logarithmic input transformations."""
    values = frame[features].to_numpy(np.float32).copy()
    values[:, 0] = np.log10(np.maximum(values[:, 0], 1e-12))
    values[:, 2:5] = np.log10(np.maximum(values[:, 2:5], 1e-12))
    return values


def fit_scaler(train_values: np.ndarray, specification: dict) -> dict[str, np.ndarray]:
    """Fit the data-dependent feature statistics on training rows only."""
    if (
        train_values.ndim != 2
        or not len(train_values)
        or not np.isfinite(train_values).all()
    ):
        raise ValueError("training features must be a finite matrix")
    tail = train_values[:, 6:]
    tail_mean = np.mean(tail, axis=0)
    tail_std = np.std(tail, axis=0, ddof=1)
    tail_std = np.where(tail_std <= 1e-12, 1.0, tail_std)
    return {
        "head_min": np.asarray(specification["feature_head_min"], dtype=np.float32),
        "head_max": np.asarray(specification["feature_head_max"], dtype=np.float32),
        "tail_mean": tail_mean.astype(np.float32),
        "tail_std": tail_std.astype(np.float32),
    }


def scale_features(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    """Scale fixed-range head features and standardized tail features."""
    head = (values[:, :6] - scaler["head_min"]) / (
        scaler["head_max"] - scaler["head_min"]
    )
    tail = (values[:, 6:] - scaler["tail_mean"]) / scaler["tail_std"]
    return np.concatenate([head, tail], axis=1).astype(np.float32)


def scale_target(cost, specification: dict) -> np.ndarray:
    """Map mean distance-computation counts to the fixed log-cost range."""
    log_cost = np.log10(np.maximum(np.asarray(cost, float), 1.0))
    return (
        (log_cost - specification["target_log10_min"])
        / (specification["target_log10_max"] - specification["target_log10_min"])
    ).astype(np.float32)


def inverse_target(values, specification: dict) -> np.ndarray:
    """Convert scaled model outputs back to distance-computation counts."""
    log_cost = (
        np.asarray(values, float)
        * (specification["target_log10_max"] - specification["target_log10_min"])
        + specification["target_log10_min"]
    )
    return np.power(10.0, log_cost)


def train_predict(
    train,
    test,
    features: list[str],
    specification: dict,
    fold_index: int,
) -> np.ndarray:
    """Train one PGTuner fold and predict held-out configurations."""
    if not len(train) or not len(test):
        raise ValueError("baseline train and test sets must be non-empty")
    seed = int(specification["seed_base"]) + int(fold_index)
    seed_everything(seed, int(specification["torch_threads_per_worker"]))
    train_values = transform_features(train, features)
    test_values = transform_features(test, features)
    scaler = fit_scaler(train_values, specification)
    x_train = torch.as_tensor(scale_features(train_values, scaler), dtype=torch.float32)
    x_test = torch.as_tensor(scale_features(test_values, scaler), dtype=torch.float32)
    y_train = torch.as_tensor(
        scale_target(train["average_search_dc_counts"], specification).reshape(-1, 1),
        dtype=torch.float32,
    )

    model = DirectPredictMLP(specification["layer_sizes"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(specification["learning_rate"]),
        weight_decay=float(specification["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(specification["scheduler_step_size"]),
        gamma=float(specification["scheduler_gamma"]),
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(specification["batch_size"])
    for _ in range(int(specification["epochs"])):
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(x_train), batch_size):
            index = order[start : start + batch_size]
            prediction = model(x_train[index])
            loss = torch.mean((prediction - y_train[index]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        scaled = model(x_test).numpy().reshape(-1)
    predicted = inverse_target(scaled, specification)
    if not np.isfinite(predicted).all():
        raise RuntimeError("baseline produced non-finite predictions")
    return predicted
