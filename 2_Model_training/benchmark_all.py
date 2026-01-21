# %%
"""
Simple benchmark script for testing sklearn models on spatial interpolation datasets.

Usage:
    uv run python 2_Model_training/benchmark_all.py

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
from sklearn.neighbors import KNeighborsRegressor  

from utils.migbt import SklearnMIXGBooster
import xgboost

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


# %%
# =============================================================================
# CONFIGURATION - Edit these to customize the benchmark
# =============================================================================

MODELS = [
#    {
#        "name": "random_forest",
#        "class": RandomForestRegressor,
#        "params": {"n_jobs": -1, "random_state": 42},
#        "search_space": {
#            "ml_model__n_estimators": [50, 100, 200],
#            "ml_model__max_features": ["sqrt", 1.0],
#            "ml_model__min_samples_leaf": [5, 10, 20],
#        },
#    },
#    {
#        "name": "random_forest_cr",
#        "class": RandomForestRegressor,
#        "number_axis": 23,
#        "params": {"n_jobs": -1, "random_state": 42},
#        "search_space": {
#            "ml_model__n_estimators": [50, 100, 200],
#            "ml_model__max_features": ["sqrt", 1.0],
#            "ml_model__min_samples_leaf": [5, 10, 20],
#        },
#    },
#    {
#        "name": "xgboost",
#        "class": xgboost.XGBRegressor,
#        "params": {
#            "n_jobs": -1,
#            "random_state": 42,
#            "subsample": 1,
#            "colsample_bytree": 1,
#            "objective": "reg:squarederror",
#            },
#        "search_space": {
#            "ml_model__max_depth": [6, 10, 15],
#            "ml_model__learning_rate": [0.05, 0.1, 0.3],
#            "ml_model__n_estimators": [100, 200],
#            "ml_model__max_bin": [1000, 2000, 5000]
#        },
        # "callbacks": [
        #     {
        #         "callback": xgboost.callback.EvaluationMonitor(period=5),
        #         "params": {
        #             "period": 5
        #         }
        #     }
        # ]
#    },
#    {
#        "name": "xgboost_cr",
#        "class": xgboost.XGBRegressor,
#        "number_axis": 23,
#        "params": {
#            "n_jobs": -1,
#            "random_state": 42,
#            "subsample": 1,
#            "colsample_bytree": 1,
#            "objective": "reg:squarederror",
#            },
#        "search_space": {
#            "ml_model__max_depth": [6, 10, 15],
#            "ml_model__learning_rate": [0.05, 0.1, 0.3],
#            "ml_model__n_estimators": [100, 200],
#            "ml_model__max_bin": [256, 1000, 2000]
#        },
        # "callbacks": [
        #     {
        #         "callback": xgboost.callback.EvaluationMonitor(period=5),
        #         "params": {
        #             "period": 5
        #         }
        #     }
        # ]
#    },
#    {
#        "name": "oblique_random_forest",
#        "class": ObliqueRandomForestRegressor,
#        "params": {
#            "random_state": 42,
#            "n_jobs": -1,
#        },
#        "search_space": {
#            "ml_model__n_estimators": [50, 100],
#            # Oblique trees often need more features per split to find good linear combinations
#            "ml_model__max_features": [1.0, "sqrt"],
#            "ml_model__max_depth": [None, 10, 20],
#            "ml_model__min_samples_leaf": [1, 5, 10],
#        },
#    },
#    {
#        "name": "geoRF",
#        "class": GeoRFRegressor,
#        "number_axis": 23,
#        "params": {
#            "n_estimators": 20,         # Number of trees
#            "max_depth": 10,            # max depth
#            "min_samples_split": 20,    # Min campioni per split
#            "max_samples": 0.5,         # Dimensione bootstrap
#            "max_features": 2,          # Features per split
#            "oob_score": False,         # Out-of-bag validation
#            "n_jobs": 1,                # Parallelizzazione
#            "random_state": 42,         # Seed per riproducibilità
#       },
#        "search_space":{},
#    },
#    {
#        "name": "kriging_tuned",
#        "class": PyKrigeWrapper,
#        "number_axis": 23,
#        "params": {
#            "coordinates_type": "euclidean",            
#        },
#        "search_space": {
#            "ml_model__variogram_model": ["spherical", "exponential", "gaussian", "linear"],
#            "ml_model__nlags": [4, 6, 10, 15],
#           "ml_model__weight": [True, False],
#            "ml_model__anisotropy_scaling": [1.0, 1.5, 2.0],
#        },
#    },
#    {
#        "name": "gam_tuned",
#        "class": PyGAMWrapper,
#        "number_axis": 23,
#        "params": {
#        },
#        "search_space": {
#            "ml_model__n_splines": [15, 25, 35, 50],
#            "ml_model__lam": [0.1, 0.6, 2.0, 10.0],
#            "ml_model__spline_order": [2, 3, 4],
#            "ml_model__use_tensor_product": [True, False],
#       },
    {
        "name": "knn_3",
        "class": KNeighborsRegressor,
        "params": {
            "n_neighbors": 3,
            "weights": "uniform",
            "algorithm": "kd_tree", 
            "leaf_size": 40,        
            "n_jobs": -1}
        },
    {
        "name": "mixgboost",
        "class": SklearnMIXGBooster,
        "params": {
            "k": 20,
            "lamb": 0.05,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 12,
            "n_jobs": -1}
        },

    {
        "name": "idw_p3",
        "class": IDWRegressor,
        "params": {
            "power": 3,
            "n_neighbors": 15
        },

    }
 ]


SIZE_SMALL = 5_000
SIZE_LARGE = 100_000
DATASETS = [
    # --- Real Datasets ---
#    {
#        "name": "bdalti",
#        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
#        "target_n": SIZE_LARGE,
#        "transform": "log"
#        },
#    {
#        "name": "bdalti_48",
#        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
#        "filter_col": "departement",
#        "filter_val": "48",
#        "target_n": SIZE_SMALL,
#        "transform": "log"
#        },
#    {
#        "name": "rgealti",
#        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
#        "target_n": SIZE_LARGE,
#        "transform": "log"
#        },
#    {
#        "name": "rgealti_48",
#        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
#        "filter_col": "departement",
#        "filter_val": "48",
#        "target_n": SIZE_SMALL,
#        "transform": "log"
#        },
    # --- Synthetic Datasets (New) ---
    {
        "name": "S-G-Sm",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Sm.parquet",
        "target_n": SIZE_SMALL,
    },
    {
        "name": "S-G-Lg",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Lg.parquet",
        "target_n": SIZE_LARGE,
    },
    {
        "name": "S-NG-Sm",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Sm.parquet",
        "target_n": SIZE_SMALL,
    },
    {
        "name": "S-NG-Lg",
        "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet",
        "target_n": SIZE_LARGE,
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
    """Load and preprocess a dataset to reach a fixed target size with strict cleaning."""
    ldf = get_df_from_s3(dataset_config["path"])
    target_n = dataset_config.get("target_n", 5_000)

    # Resolve schema and rename if necessary
    columns = ldf.collect_schema().names()
    if "val" in columns:
        ldf = ldf.rename({"val": "value"})

    # 1. Filters (Predicate Pushdown)
    if "filter_col" in dataset_config:
        print(f"  Filtering {dataset_config['filter_col']} = {dataset_config['filter_val']}...")
        ldf = ldf.filter(pl.col(dataset_config["filter_col"]) == str(dataset_config["filter_val"]))

    # 2. Initial Cleaning (Remove nulls and non-positive values for log)
    ldf = ldf.filter(
        (pl.col("value").is_not_null()) & 
        (pl.col("value") > 0) &
        (pl.col("x").is_not_null()) &
        (pl.col("y").is_not_null())
    )

    # 3. Optimized Data Fetching (Limit for massive files)
    if "rgealti" in dataset_config["name"]:
        fetch_limit = target_n # Buffer for cleaning
        print(f"  RGEALTI/Large file detected: Pre-fetching {fetch_limit} rows...")
        ldf = ldf.head(fetch_limit)

    # 4. Collection to RAM
    print(f"  Collecting and sampling to exactly {target_n} rows...")
    df = ldf.select(["x", "y", "value"]).collect()

    # 5. Log Transform
    if dataset_config.get("transform") == "log":
        print("  Applying log transformation...")
        df = df.with_columns(pl.col("value").log())

    # 6. CRITICAL: Final Finite Check
    # This removes any NaNs or Infs created by the log transform or existing in data
    df = df.filter(
        pl.col("value").is_finite() & 
        pl.col("x").is_finite() & 
        pl.col("y").is_finite()
    )

    # 7. Exact Sampling
    if len(df) > target_n:
        df = df.sample(n=target_n, seed=RANDOM_STATE)
    
    print(f"  Final clean dataset size: {len(df)}")

    X = df.select(["x", "y"])
    y = df.select("value").to_numpy().ravel()
    
    del df
    gc.collect()

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_model(model_config: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate models, handling GeoRF, CV search, and direct fits."""
    
    # 1. Setup Common Components
    # We use .copy() to avoid modifying the original config if we need to pop keys
    params = model_config.get("params", {}).copy()
    
    # Safety: Kriging/GAM wrappers often don't support n_jobs in __init__
    # If it causes issues, you can explicitly pop it here:
    # params.pop("n_jobs", None) 

    model_instance = model_config["class"](**params)
    number_axis = model_config.get("number_axis", 1)
    
    # Identifiers for logic branches
    is_georf = "geoRF" in model_config["name"]
    search_space = model_config.get("search_space", {})
    has_search_space = len(search_space) > 0

    # 2. Build Pipeline
    pipeline = Pipeline([
        ("coord_rotation", AddCoordinatesRotation(
            coordinates_names=("x", "y"),
            number_axis=number_axis
        )),
        ("pandas_converter", ConvertToPandas()),
        ("ml_model", model_instance),
    ])

    start = time.perf_counter()

    # =========================================================================
    # BRANCH 1: GeoRF (Direct fit, special parameters)
    # =========================================================================
    if is_georf:
        print(f"    [GeoRF Mode] Skipping CV, fitting directly...")
        pipeline.fit(X_train, y_train, ml_model__gens="da")
        best_pipeline = pipeline
        best_params = params

    # =========================================================================
    # BRANCH 2: Models with Search Space (RandomizedSearchCV)
    # =========================================================================
    elif has_search_space:
        print(f"    [Search Mode] Running RandomizedSearchCV...")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=N_ITER_SEARCH,
            cv=CV_FOLDS,
            scoring='r2',
            verbose=1,
            n_jobs=-1  # Parallelize folds
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_

    # =========================================================================
    # BRANCH 3: Standard Fit (No CV, no special params)
    # =========================================================================
    else:
        print(f"    [Standard Mode] Fitting pipeline directly...")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = params

    training_time = time.perf_counter() - start

    # 3. Evaluate
    y_pred = best_pipeline.predict(X_test)

    # Compute metrics
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}

    return {
        "model": model_config["name"],
        "best_params": best_params,
        "training_time": round(training_time, 2),
        **metrics,
    }


def run_benchmark(models: list, datasets: list) -> dict:
    """Run all model/dataset combinations with error handling."""
    results = []

    for dataset in datasets:
        print(f"\n{'='*50}\nLoading dataset: {dataset['name']}\n{'='*50}")
        try:
            X_train, X_test, y_train, y_test = load_dataset(dataset)
            print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
        except Exception as e:
            print(f"  FAILED to load dataset {dataset['name']}: {e}")
            continue

        for model in models:
            print(f"  Training: {model['name']}")
            try:
                result = run_model(model, X_train, X_test, y_train, y_test)
                result["dataset"] = dataset["name"]
                results.append(result)
                print(f"    R2: {result['r2_score']:.4f}, Time: {result['training_time']}s")
            except Exception as e:
                # This ensures one model failing doesn't stop the whole benchmark
                print(f"    ERROR training {model['name']}: {type(e).__name__}: {e}")
                continue

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "coord_rotation_axis": COORD_ROTATION_AXIS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "results": results,
    }


def run_benchmark(models: list, datasets: list) -> dict:
    """Run all model/dataset combinations with error handling."""
    results = []

    for dataset in datasets:
        print(f"\n{'='*50}\nLoading dataset: {dataset['name']}\n{'='*50}")
        try:
            X_train, X_test, y_train, y_test = load_dataset(dataset)
            print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
        except Exception as e:
            print(f"  FAILED to load dataset {dataset['name']}: {e}")
            continue

        for model in models:
            print(f"  Training: {model['name']}")
            try:
                result = run_model(model, X_train, X_test, y_train, y_test)
                result["dataset"] = dataset["name"]
                results.append(result)
                print(f"    R2: {result['r2_score']:.4f} | RMSE: {result['rmse']:.4f} | MAE: {result['mae']:.4f} | Time: {result['training_time']}s")

            except Exception as e:
                # This ensures one model failing doesn't stop the whole benchmark
                print(f"    ERROR training {model['name']}: {type(e).__name__}: {e}")
                continue

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
