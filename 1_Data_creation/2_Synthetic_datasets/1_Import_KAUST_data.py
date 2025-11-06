# %%
import os
import glob
import re
import polars as pl
import shutil
import requests
from osgeo import gdal
import s3fs
import zipfile

# %%
# Connect to S3
fs = s3fs.S3FileSystem(anon=False)  # set anon=True for public buckets

# Define your S3 path
s3_path = 's3://projet-benchmark-spatial-interpolation'

# %%
# Download data
# mc cp s3/projet-benchmark-spatial-interpolation/data/synthetic/second_kaust_competition_data.zip ./
# %%

with zipfile.ZipFile("/home/onyxia/µwork/second_kaust_competition_data.zip", 'r') as zip_ref:
    zip_ref.extractall(path='rawdata/')
# %%
list_files = glob.glob('rawdata/**/*.csv', recursive=True)
list_files
# %%

for file in list_files:
    print(file)
    print("    Read data")
    df = pl.read_csv(file)
    filename = os.path.splitext(os.path.basename(file))[0]

    # Write to S3
    print("    Write to S3")
    df.write_parquet(
        f"{s3_path}/data/synthetic/KAUST2/{filename}.parquet",
        use_pyarrow=True
    )

# %%