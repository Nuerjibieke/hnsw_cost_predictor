"""Train and evaluate the paper's fixed query-cost regressor."""

from __future__ import annotations

import numpy as np


def joint_holdout_masks(dataset_values, model_values, held_dataset, held_model):
    """Return strict LODO+LOMO train and test masks."""
    datasets = np.asarray(dataset_values).astype(str)
    models = np.asarray(model_values).astype(str)
    train = (datasets != str(held_dataset)) & (models != str(held_model))
    test = (datasets == str(held_dataset)) & (models == str(held_model))
    return train, test


def build_lgbm(fold_index: int, n_features: int, seed_base: int):
    """Create the fixed LightGBM regressor for one holdout fold."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=420,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.9,
        subsample_freq=0,
        colsample_bytree=0.9,
        reg_lambda=4.0,
        random_state=int(seed_base) + int(fold_index),
        n_jobs=4,
        verbosity=-1,
        force_col_wise=True,
        interaction_constraints=[list(range(n_features))],
    )


def qerror(true_cost, pred_cost):
    """Compute multiplicative Q-error values."""
    true_cost = np.maximum(np.asarray(true_cost, float), 1.0)
    pred_cost = np.maximum(np.asarray(pred_cost, float), 1.0)
    return np.maximum(true_cost / pred_cost, pred_cost / true_cost)


def metric(true_log, pred_log):
    """Compute the prediction metrics reported in the paper."""
    true_log = np.asarray(true_log, float)
    pred_log = np.asarray(pred_log, float)
    true_cost = np.maximum(np.expm1(true_log), 1.0)
    pred_cost = np.maximum(np.expm1(pred_log), 1.0)
    errors = qerror(true_cost, pred_cost)
    return {
        "n_rows": len(true_log),
        "mae_log": float(np.mean(np.abs(true_log - pred_log))),
        "qerror_median": float(np.nanmedian(errors)),
        "qerror_p95": float(np.nanpercentile(errors, 95)),
        "acc_1p2x": float(np.nanmean(errors <= 1.2)),
        "acc_1p5x": float(np.nanmean(errors <= 1.5)),
    }
