
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def project_path(*parts):
    return os.path.join(ROOT, *parts)

def test_processed_dataset_exists():
    train = pd.read_csv(project_path("data", "processed", "train.csv"))
    test = pd.read_csv(project_path("data", "processed", "test.csv"))

    assert len(train) > 0
    assert len(test) > 0
    assert "text" in train.columns
    assert "label" in train.columns

def test_expected_labels():
    train = pd.read_csv(project_path("data", "processed", "train.csv"))
    labels = train["label"].unique()

    assert len(labels) == 8

def test_benchmark_results():
    results = pd.read_csv(project_path("benchmark_results.csv"))

    assert {"Model", "Size MB", "Mean latency ms"}.issubset(results.columns)
    assert "Original" in results["Model"].values
    assert "Dynamic INT8" in results["Model"].values
    assert (results["Size MB"] > 0).all()
    assert (results["Mean latency ms"] > 0).all()

def test_quantized_model_is_smaller():
    results = pd.read_csv(project_path("benchmark_results.csv"))

    original = results.loc[
        results["Model"] == "Original", "Size MB"
    ].iloc[0]

    quantized = results.loc[
        results["Model"] == "Dynamic INT8", "Size MB"
    ].iloc[0]

    assert quantized < original

def test_quantized_weights_exist():
    weights = project_path(
        "models", "dynamic_quantized", "quantized_state_dict.pt"
    )

    assert os.path.exists(weights)
    assert os.path.getsize(weights) > 0
