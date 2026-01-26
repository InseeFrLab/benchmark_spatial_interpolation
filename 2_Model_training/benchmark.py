"""
Simple benchmark script for testing sklearn models on spatial interpolation datasets.

Usage:
    uv run python 2_Model_training/benchmark.py

Extensibility:
    - Add models: append to MODELS list
    - Add datasets: append to DATASETS list
    - Add metrics: append to METRICS list
"""
import json
import time
import gc
from datetime import datetime
from pathlib import Path
import polars as pl
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import xgboost
from utils.migbt import SklearnMIXGBooster
from utils.kriging_wrapper import PyKrigeWrapper
from utils.gam_wrapper import PyGAMWrapper
import sys
sys.path.append('geoRF')
from geoRF import GeoRFRegressor
from utils.idw import IDWRegressor
from utils.functions import AddCoordinatesRotation, ConvertToPandas
from utils.s3 import get_df_from_s3
from utils.patch_treeple import apply_treeple_patch
apply_treeple_patch()
from treeple.ensemble import ObliqueRandomForestRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================


MODELS = [
    # --- Random Forest ---
    {
        "name": "random_forest",
        "class": RandomForestRegressor,
        "number_axis": 1,
        "params": {
            "n_estimators": 250,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42
            },
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "params": {"n_estimators": 250, "max_features": "sqrt", "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42},
    },

    # --- XGBoost ---
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "number_axis": 1,
        "params": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 42, "objective": "reg:squarederror"},
    },
    {
        "name": "xgboost_cr",
        "class": xgboost.XGBRegressor,
        "number_axis": 23,
        "params": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 42, "objective": "reg:squarederror"},
    },

    # --- MIXGBoost (Spatially Weighted) ---
    {
        "name": "mixgboost",
        "class": SklearnMIXGBooster,
        "number_axis": 1,
        "params": {
            "k": 20,
            "lamb": 0.05,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 12,
            "n_jobs": -1
        }
    },
    {
        "name": "mixgboost_cr",
        "class": SklearnMIXGBooster,
        "number_axis": 23,
        "params": {
            "k": 20,
            "lamb": 0.05,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 12,
            "n_jobs": -1
        }
    },

    # --- Oblique Random Forest ---
    {
        "name": "oblique_rf",
        "class": ObliqueRandomForestRegressor,
        "number_axis": 1,
        "params": {"n_estimators": 100, "max_features": 1.0, "max_depth": 20, "random_state": 42, "n_jobs": -1},
    },
    # Very Low interest of cr, already look for non-orthogonal splits

    # --- GeoRF ---
    {
        "name": "geoRF",
        "class": GeoRFRegressor,
        "number_axis": 1,
        "params": {"n_estimators": 50, "max_depth": 15, "min_samples_split": 10, "max_features": 2, "n_jobs": 1, "random_state": 42},
    },
    # Very low interest of cr, already look for non-orthogonal splits

    # --- Kriging ---
    {
        "name": "kriging",
        "class": PyKrigeWrapper,
        "number_axis": 1,
        "params": {"variogram_model": "exponential", "nlags": 10, "weight": True, "coordinates_type": "euclidean"},
    },
    # No point of cr, euclidian distance invariant to cr

    # --- GAM ---
    {
        "name": "gam",
        "class": PyGAMWrapper,
        "number_axis": 1,
        "params": {"n_splines": 25, "lam": 0.6, "spline_order": 3},
    },
    {
        "name": "gam_cr",
        "class": PyGAMWrapper,
        "number_axis": 23,
        "params": {"n_splines": 25, "lam": 0.6, "spline_order": 3},
    },

    # --- Baselines ---
    {
        "name": "knn_3",
        "class": KNeighborsRegressor,
        "number_axis": 1,
        "params": {"n_neighbors": 3, "weights": "distance", "n_jobs": -1}
    },
    # No point of cr, euclidian distance invariant to cr

    {
        "name": "idw_p3",
        "class": IDWRegressor,
        "number_axis": 1,
        "params": {"power": 3, "n_neighbors": 15},
    }
    # No point of cr, euclidian distance invariant to cr
]

SIZE_SMALL = 5_000
SIZE_LARGE = 100_000
DATASETS = [
    # --- Real Datasets ---
    {
        "name": "rgealti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "target_n": SIZE_LARGE,
        "transform": "log"
        },
    {
        "name": "bdalti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "target_n": SIZE_LARGE,
        "transform": "log"
        },
    {
        "name": "bdalti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "target_n": SIZE_SMALL,
        "transform": "log"
        },
    {
        "name": "rgealti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "target_n": SIZE_SMALL,
        "transform": "log"
        },
    # --- Synthetic Datasets (New) ---
    {"name": "S-G-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Sm.parquet", "target_n": SIZE_SMALL},
    {"name": "S-G-Lg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Lg.parquet", "target_n": SIZE_LARGE},
    {"name": "S-NG-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Sm.parquet", "target_n": SIZE_SMALL},
    {"name": "S-NG-Lg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet", "target_n": SIZE_LARGE},
]

METRICS = [
    {"name": "r2_score", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error},
]

COORD_ROTATION_AXIS = 23
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def load_dataset(dataset_config: dict) -> tuple:
    """Load and preprocess massive datasets with a larger safety slice."""
    ldf = get_df_from_s3(dataset_config["path"])
    target_n = dataset_config.get("target_n", 5_000)

    # Resolve schema
    if "val" in ldf.collect_schema().names():
        ldf = ldf.rename({"val": "value"})

    # 1. Logic for Massive Datasets
    if "rgealti" in dataset_config["name"]:
        # We take a 1M row slice to ensure that after all cleaning filters, 
        # we still have at least 100k valid rows to sample from.
        print(f"  Massive dataset: Slicing first 1,000,000 rows lazily...")
        ldf = ldf.slice(0, 1_000_000)

    # 2. Filtering and Cleaning (Lazy)
    ldf = ldf.filter(
        (pl.col("value").is_not_null()) &
        (pl.col("value") > 0) &
        (pl.col("x").is_not_null()) &
        (pl.col("y").is_not_null())
    )

    # 3. Collection (Updated engine parameter)
    print(f"  Streaming data collection to RAM...")
    # 'streaming=True' is now 'engine="streaming"' in newer Polars versions
    df = ldf.select(["x", "y", "value"]).collect(engine="streaming")

    # 4. Log Transform and Finite Check
    if dataset_config.get("transform") == "log":
        df = df.with_columns(pl.col("value").log())

    df = df.filter(
        pl.col("value").is_finite() &
        pl.col("x").is_finite() &
        pl.col("y").is_finite()
    )

    # 5. Exact Sampling
    if len(df) >= target_n:
        df = df.sample(n=target_n, seed=RANDOM_STATE)
    else:
        print(f"  WARNING: Only {len(df)} rows available after cleaning.")

    print(f"  Final clean dataset size: {len(df)}")

    X = df.select(["x", "y"])
    y = df.select("value").to_numpy().ravel().astype(float)

    del df
    gc.collect()

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_model(model_config: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate models, skipping GeoRF on large datasets for efficiency."""

    params = model_config.get("params", {}).copy()
    model_name = model_config["name"]
    avoid_big_set = ("geoRF" in model_name) or ("kriging" in model_name) or ("gam" in model_name)

    # --- Logic: Skip for slow algorithm if dataset is too large ---
    LARGE_THRESHOLD = 10_000
    if avoid_big_set and len(X_train) > LARGE_THRESHOLD:
        print(f"    [SKIP] {model_name}: Dataset too large ({len(X_train)} rows).")
        return None

    model_instance = model_config["class"](**params)
    number_axis = model_config.get("number_axis", 1)

    pipeline = Pipeline([
        ("coord_rotation", AddCoordinatesRotation(
            coordinates_names=("x", "y"),
            number_axis=number_axis
        )),
        ("pandas_converter", ConvertToPandas()),
        ("ml_model", model_instance),
    ])

    start = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = time.perf_counter() - start
    y_pred = pipeline.predict(X_test)

    # Compute metrics
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}

    return {     
        "model": model_name,
        "best_params": params,
        "training_time": round(training_time, 2),
        **metrics,
    }


def run_benchmark(models: list, datasets: list) -> dict:
    results = []
    for dataset in datasets:
        print(f"\nProcessing: {dataset['name']}")
        try:
            X_train, X_test, y_train, y_test = load_dataset(dataset)
        except Exception as e:
            print(f"  FAILED load {dataset['name']}: {e}")
            continue

        for model in models:
            print(f"  Training: {model['name']}...")
            try:
                result = run_model(model, X_train, X_test, y_train, y_test)
                if result is not None:
                    result["dataset"] = dataset["name"]
                    results.append(result)
                    print(f"    R2: {result['r2_score']:.4f} | Time: {result['training_time']}s")
            except Exception as e:
                print(f"    ERROR training {model['name']}: {e}")
                continue

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "results": results,
    }


def save_results(results: dict, output_dir: str = "results") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    filepath = output_path / "benchmark_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    benchmark_results = run_benchmark(MODELS, DATASETS)
    save_results(benchmark_results)
