import numpy as np
import pandas as pd

from semiyield.benchmark import (
    BenchmarkConfig,
    _evaluate_split,
    run_benchmark,
    select_budget_threshold,
    select_threshold,
)
from semiyield.data import SecomDataset


def test_threshold_uses_cost_tradeoff():
    threshold = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9], missed_cost=10, review_cost=1)
    assert 0.2 < threshold <= 0.6


def test_budget_threshold_selects_top_fraction():
    scores = np.arange(10) / 10
    threshold = select_budget_threshold(scores, 0.2)
    assert (scores >= threshold).sum() >= 2
    assert (scores > threshold).sum() <= 2


def test_benchmark_writes_reproducible_artifacts(sample_data, tmp_path):
    frame, target = sample_data
    dataset = SecomDataset(
        frame,
        target,
        pd.Series(pd.date_range("2024-01-01", periods=len(frame), freq="h")),
        {"name": "fixture"},
    )
    result = run_benchmark(
        dataset,
        config=BenchmarkConfig(models=("dummy", "logistic"), folds=2, repeats=1),
        output_dir=tmp_path,
    )
    assert set(result["summary"]["model"]) == {"dummy", "logistic"}
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "fold_metrics.csv").exists()
    assert np.isfinite(result["summary"]["pr_auc_mean"]).all()
    assert result["manifest"]["resource_guard"] == "not_applied_python_api"


def test_calibrated_oof_threshold_stays_on_probability_scale(sample_data):
    frame, target = sample_data
    train_idx = np.arange(0, 30)
    test_idx = np.arange(30, len(frame))
    result = _evaluate_split(
        frame,
        target,
        train_idx,
        test_idx,
        "logistic",
        BenchmarkConfig(models=("logistic",), folds=2, repeats=1, calibration_folds=2),
    )
    assert 0.0 <= result["threshold"] <= 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
