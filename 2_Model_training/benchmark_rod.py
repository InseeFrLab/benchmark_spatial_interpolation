# %%
"""
Simple benchmark script for testing sklearn models on spatial interpolation datasets.

Usage:
    uv run python 2_Model_training/benchmark_rod.py

Extensibility:
    - Add models: append to MODELS list
    - Add datasets: append to DATASETS list
    - Add metrics: append to METRICS list
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from polars import col as c

# Apply Intel sklearn acceleration
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import xgboost

from utils.functions import AddCoordinatesRotation, ConvertToPandas
from utils.s3 import get_df_from_s3
from utils.patch_treeple import apply_treeple_patch
apply_treeple_patch()

from treeple.ensemble import ObliqueRandomForestRegressor


# %%
# =============================================================================
# CONFIGURATION - Edit these to customize the benchmark
# =============================================================================

MODELS = [
    {
        "name": "random_forest",
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 50,
            "max_features": "sqrt",
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 5
        },
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "params": {
            "n_estimators": 50,
            "max_features": "sqrt",
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 5
        },
    },
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "learning_rate": 0.3,
            "subsample": 1,
            "colsample_bytree": 1,
            "objective": "reg:squarederror",
            "random_state": 42,
            "max_bin": 2000,
            "n_jobs": -1,
        },
        # "callbacks": [
        #     {
        #         "callback": xgboost.callback.EvaluationMonitor(period=5),
        #         "params": {
        #             "period": 5
        #         }
        #     }
        # ]
    },
    {
        "name": "xgboost_cr",
        "class": xgboost.XGBRegressor,
        "number_axis": 23,
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "learning_rate": 0.3,
            "subsample": 1,
            "colsample_bytree": 1,
            "objective": "reg:squarederror",
            "random_state": 42,
            "max_bin": 256,
            "n_jobs": -1,
        },
        # "callbacks": [
        #     {
        #         "callback": xgboost.callback.EvaluationMonitor(period=5),
        #         "params": {
        #             "period": 5
        #         }
        #     }
        # ]
    },
    {
        "name": "oblique_random_forest",
        "class": ObliqueRandomForestRegressor,
        "params": {
            "n_estimators": 50,
            "max_features": 1.0, # Oblique trees often benefit from seeing all features per split
            "random_state": 42,
            "n_jobs": -1,
        },
    },
]

DATASETS = [
    # --- SYNTHETIC DATASETS ---
    {"name": "S-G-Sm",  "path": "s3://.../S-G-Sm.parquet",  "sample": 1.0, "type": "synthetic"},
    {"name": "S-G-Lg",  "path": "s3://.../S-G-Lg.parquet",  "sample": 0.1, "type": "synthetic"},
    {"name": "S-NG-Sm", "path": "s3://.../S-NG-Sm.parquet", "sample": 1.0, "type": "synthetic"},
    {"name": "S-NG-Lg", "path": "s3://.../S-NG-Lg.parquet", "sample": 0.1, "type": "synthetic"},

    # --- REAL DATASETS (Using BDALTI 25m for Small, RGEALTI 5m for Large) ---
    
    # 5. Real - Grid - Small (Full department at 25m)
    {
        "name": "R-G-Sm", 
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement", "filter_val": "48", "sample": 1.0, "transform": "log"
    },
    # 6. Real - Grid - Large (Full department at 5m - Very Dense)
    {
        "name": "R-G-Lg", 
        "path": "s3://projet-benchmark-spatial-interpolation/RGEALTI/RGEALTI_parquet/",
        "filter_col": "departement", "filter_val": "48", "sample": 0.5, "transform": "log"
    },
    # 7. Real - Non-Grid - Small (Sparse random points from 25m)
    {
        "name": "R-NG-Sm", 
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement", "filter_val": "48", "sample": 5000, "transform": "log"
    },
    # 8. Real - Non-Grid - Large (Scattered random points from 5m)
    {
        "name": "R-NG-Lg", 
        "path": "s3://projet-benchmark-spatial-interpolation/RGEALTI/RGEALTI_parquet/",
        "filter_col": "departement", "filter_val": "48", "sample": 200000, "transform": "log"
    },
]

METRICS = [
    {"name": "r2_score", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error},
]

# Pipeline settings
COORD_ROTATION_AXIS = 23
TEST_SIZE = 0.5
RANDOM_STATE = 123456


# %%
# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_dataset(dataset_config: dict) -> tuple:
    """Load and preprocess a dataset based on size, grid/non-grid, and nature."""
    print(f"  Fetching data from: {dataset_config['path']}")
    ldf = get_df_from_s3(dataset_config["path"])

    # Standardize column names
    columns = ldf.collect_schema().names()
    if "val" in columns and "value" not in columns:
        ldf = ldf.rename({"val": "value"})

    # Apply filters (e.g., selecting a specific department)
    if "filter_col" in dataset_config:
        ldf = ldf.filter(pl.col(dataset_config["filter_col"]) == dataset_config["filter_val"])

    # Clean data: Remove NaNs and non-positive values for log transform safety
    ldf = ldf.filter(
        (pl.col("value").is_not_null()) & 
        (~pl.col("value").is_nan()) & 
        (pl.col("value") > 0)
    )

    # Collect into memory for sampling/splitting
    df = ldf.select("x", "y", "value").collect()

    # Handle Sampling (Supports both % fraction and absolute N points)
    sample_val = dataset_config.get("sample", 1.0)
    if isinstance(sample_val, float) and sample_val < 1.0:
        df = df.sample(fraction=sample_val, seed=RANDOM_STATE)
    elif isinstance(sample_val, int):
        # Ensure we don't try to sample more points than exist
        n_samples = min(sample_val, df.height)
        df = df.sample(n=n_samples, seed=RANDOM_STATE)

    # Separate target and features
    X = df.select("x", "y")
    y = df.select("value").to_numpy().ravel()

    # Transform the target if requested
    if dataset_config.get("transform") == "log":
        y = np.log(y)

    print(f"  Dataset loaded: {len(y)} points.")
    
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_model(model_config: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a single model."""

    # Create model instance
    model = model_config["class"](**model_config["params"])

    # Detect if there is coordinate rotation
    if "number_axis" in model_config:
        number_axis = model_config["number_axis"]
    else:
        number_axis = 1

    # Build the pipeline
    pipeline = Pipeline(
        [
            ("coord_rotation", AddCoordinatesRotation(
                coordinates_names=("x", "y"),
                number_axis=number_axis
            )),
            ("pandas_converter", ConvertToPandas()),
            ("ml_model", model),
        ]
    )

    # Train with timing
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # Predict
    y_pred = pipeline.predict(X_test)

    # Compute metrics
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}

    return {
        "model": model_config["name"],
        "params": model_config["params"],
        "training_time": round(training_time, 2),
        **metrics,
    }


def run_benchmark(models: list, datasets: list) -> dict:
    """Run all model/dataset combinations."""
    results = []

    for dataset in datasets:
        print(f"Loading dataset: {dataset['name']}")
        X_train, X_test, y_train, y_test = load_dataset(dataset)
        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

        for model in models:
            print(f"  Training: {model['name']}")
            result = run_model(model, X_train, X_test, y_train, y_test)
            result["dataset"] = dataset["name"]
            results.append(result)
            print(f"    R2: {result['r2_score']:.4f}, Time: {result['training_time']}s")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "coord_rotation_axis": COORD_ROTATION_AXIS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "results": results,
    }


def save_results(results: dict, output_dir: str = "results") -> Path:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = results["timestamp"][:10]
    filepath = output_path / f"benchmark_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


# %%
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_benchmark(MODELS, DATASETS)
    save_results(results)

# %%
