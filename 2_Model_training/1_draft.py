# %%
import os
import s3fs
from utils import s3

# Modules for data manipulation
import polars as pl
from polars import col as c
import numpy as np

# Modules for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# %%
# Connect to S3
fs = s3fs.S3FileSystem(anon=False)  # set anon=True for public buckets

# Define your S3 path
datapath = 's3://projet-benchmark-spatial-interpolation/data'

# %%
temp = s3.get_df_from_s3(f"{datapath}/real/RGEALTI/RGEALTI_parquet/")

