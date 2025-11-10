# %%
import os
import s3fs
from utils import s3
import time

# Modules for data manipulation
import polars as pl
from polars import col as c
import numpy as np

# Module for plots
import matplotlib.pyplot as plt

# Module to speed up scikit-learn
from sklearnex import patch_sklearn
# patch_sklearn()

# Modules for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils.functions import AddCoordinatesRotation

# %%
# Connect to S3
fs = s3fs.S3FileSystem(anon=False)  # set anon=True for public buckets

# Define your S3 path
datapath = 's3://projet-benchmark-spatial-interpolation/data'

# %%
# Extract one departement
data = (
    s3.get_df_from_s3(f"{datapath}/real/BDALTI/BDALTI_parquet/")
    .filter(c.departement == "48")
    .filter(~c.value.is_nan())
    .select("x", "y", "value")
    .collect()
)
# %%
# Plot the data
# Convert to pandas DataFrame
# Pivot the data to create a 2D grid for raster plotting
raster = data.to_pandas().pivot(index='y', columns='x', values='value')

# Plot raster with imshow
plt.imshow(raster.values, origin='lower', interpolation='none', cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Raster plot from Polars dataframe')
plt.show()
# %%
# Separate target and features
X = data.select("x", "y")
y = data.select("value").to_numpy().ravel()

# %%
# Perform a train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# %%
# Instantiate the model
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

full_model = Pipeline(
    [
        ("coord_rotation", AddCoordinatesRotation()),
        ("ml_model", rf_model)
    ]
)
# %%
# Define model hyperparameters
parameters_model = {
    "coord_rotation__coordinates_names": ("x", "y"),
    "coord_rotation__number_axis": 23,
    "ml_model__n_estimators": 50,
    "ml_model__max_features": "sqrt",
    "ml_model__min_samples_split": 40,
    "ml_model__min_samples_leaf": 20,
    "ml_model__verbose": 3,
    "ml_model__oob_score": True
}

full_model.set_params(**parameters_model)

# %%
# Entraînement du modèle
start_time = time.perf_counter()
full_model.fit(X_train, y_train)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f} secondes")

# %%
# Measure performance

# Specific to random forest: out-of-bag performance
print(rf_model.oob_score_)

# For all models: performance on the test set
y_pred = full_model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print(r2_test)
# %%
