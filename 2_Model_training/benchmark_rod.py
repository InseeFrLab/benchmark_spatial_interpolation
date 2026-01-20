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
import gc  # Added for manual memory cleanup
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from polars import col as c

# Apply Intel sklearn acceleration
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

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
        "params": {"n_jobs": -1, "random_state": 42},
        "search_space": {
            "ml_model__n_estimators": [50, 100, 200],
            "ml_model__max_features": ["sqrt", 1.0],
            "ml_model__min_samples_leaf": [5, 10, 20],
        },
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "params": {"n_jobs": -1, "random_state": 42},
        "search_space": {
            "ml_model__n_estimators": [50, 100, 200],
            "ml_model__max_features": ["sqrt", 1.0],
            "ml_model__min_samples_leaf": [5, 10, 20],
        },
    },
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "params": {
            "n_jobs": -1,
            "random_state": 42,
            "subsample": 1,
            "colsample_bytree": 1,
            "objective": "reg:squarederror",
            },
        "search_space": {
            "ml_model__max_depth": [6, 10, 15],
            "ml_model__learning_rate": [0.05, 0.1, 0.3],
            "ml_model__n_estimators": [100, 200],
            "ml_model__max_bin": [1000, 2000, 5000]
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
            "n_jobs": -1,
            "random_state": 42,
            "subsample": 1,
            "colsample_bytree": 1,
            "objective": "reg:squarederror",
            },
        "search_space": {
            "ml_model__max_depth": [6, 10, 15],
            "ml_model__learning_rate": [0.05, 0.1, 0.3],
            "ml_model__n_estimators": [100, 200],
            "ml_model__max_bin": [256, 1000, 2000]
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
            "random_state": 42,
            "n_jobs": -1,
        },
        "search_space": {
            "ml_model__n_estimators": [50, 100],
            # Oblique trees often need more features per split to find good linear combinations
            "ml_model__max_features": [1.0, "sqrt"],
            "ml_model__max_depth": [None, 10, 20],
            "ml_model__min_samples_leaf": [1, 5, 10],
        },
    },
]

DATASETS = [
    # --- Real Datasets ---
    {
        "name": "bdalti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "sample": 0.005,
        "transform": "log"
        },
    {
        "name": "bdalti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "sample": 0.4,
        "transform": "log"
        },
    {
        "name": "rgealti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "sample": 0.00001,
        "transform": "log"
        },
    {
        "name": "rgealti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "sample": 0.004,
        "transform": "log"
        },
    # --- Synthetic Datasets (New) ---
    {
        "name": "S-G-Sm",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Sm.parquet",
        "sample": 1.0,  # Use all points for small sets
    },
    {
        "name": "S-G-Lg",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Lg.parquet",
        "sample": 0.1,  # 1M points is heavy, sample 100k for faster benchmarking
    },
    {
        "name": "S-NG-Sm",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Sm.parquet",
        "sample": 1.0,
    },
    {
        "name": "S-NG-Lg",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet",
        "sample": 0.1,
    },
]

METRICS = [
    {"name": "r2_score", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error},
]

# Pipeline settings
COORD_ROTATION_AXIS = 23
N_ITER_SEARCH = 5
CV_FOLDS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 123456


# %%
# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_dataset(dataset_config: dict) -> tuple:
    """Load and preprocess a dataset using fixed row limits for heavy data."""
    ldf = get_df_from_s3(dataset_config["path"])

    # Resolve schema and rename if necessary
    columns = ldf.collect_schema().names()
    if "val" in columns:
        ldf = ldf.rename({"val": "value"})

    # 1. Filters (Predicate Pushdown)
    if "filter_col" in dataset_config:
        print(f"  Filtering {dataset_config['filter_col']} = {dataset_config['filter_val']}...")
        ldf = ldf.filter(pl.col(dataset_config["filter_col"]) == str(dataset_config["filter_val"]))

    # 2. Optimized Sampling/Limiting
    # For RGEALTI, we use .head() to avoid scanning 22 billion rows
    if "rgealti" in dataset_config["name"]:
        # Let's start safe with 100k points. Increase if stable.
        n_points = 100_000
        print(f"  RGEALTI detected: Limiting to {n_points} rows via .head()...")
        ldf = ldf.head(n_points)
    elif "sample" in dataset_config:
        smpl = dataset_config["sample"]
        print(f"  Sampling with fraction {smpl}...")
        ldf = ldf.filter(pl.struct(["x"]).map_batches(
            lambda s: pl.Series(np.random.random(s.len()) < smpl)
        ))

    # Clean data (ensure value > 0 for log transform)
    ldf = ldf.filter(
        (pl.col("value").is_not_null()) &
        (pl.col("value").is_not_nan()) &
        (pl.col("value") > 0)
    )

    # 3. Log Transform
    if dataset_config.get("transform") == "log":
        print("  Applying log transformation...")
        ldf = ldf.with_columns(pl.col("value").log())

    # 4. Final Collection
    print("  Collecting into RAM...")
    df = ldf.select(["x", "y", "value"]).collect()

    X = df.select(["x", "y"])
    y = df.select("value").to_numpy().ravel()

    del df
    gc.collect()

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_model(model_config: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train with Hyperparameter Optimization and evaluate."""

    # 1. Basic Setup
    model_instance = model_config["class"](**model_config["params"])
    number_axis = model_config.get("number_axis", 1)

    # 2. Build Pipeline
    pipeline = Pipeline([
        ("coord_rotation", AddCoordinatesRotation(
            coordinates_names=("x", "y"),
            number_axis=number_axis
        )),
        ("pandas_converter", ConvertToPandas()),
        ("ml_model", model_instance),
    ])

    # 3. Setup Random Search
    # This treats the entire pipeline as the estimator
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=model_config["search_space"],
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring='r2',
        verbose=1,
        n_jobs=-1  # Parallelize the CV folds
    )

    # 4. Fit & Time
    start = time.perf_counter()
    search.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # 5. Evaluate Best Model
    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    # Compute metrics
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}

    return {
        "model": model_config["name"],
        "best_params": search.best_params_,  # Save what was actually chosen
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
