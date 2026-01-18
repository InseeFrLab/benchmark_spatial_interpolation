import polars as pl
import os
import joblib

# Connection information for Polars
storage_options = {
    "aws_endpoint":  "https://"+os.environ['AWS_S3_ENDPOINT'],
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "aws_region": os.environ["AWS_DEFAULT_REGION"],
    "aws_token": os.environ["AWS_SESSION_TOKEN"]
}


def get_df_from_s3(s3_path: str):
    ldf = pl.scan_parquet(
        source=f'{s3_path}',
        storage_options=storage_options
    )
    # Check S3 access
    ldf.collect_schema()
    return ldf


def write_df_to_s3(df, s3_path: str, partition_cols=None):
    # Save data to S3
    if partition_cols is None:
        with conf_s3.fs.open(s3_path, 'wb') as f:
            df.write_parquet(f, use_pyarrow=True)
    else:
        with conf_s3.fs.open(s3_path, 'wb') as f:
            df.write_parquet(
                f,
                use_pyarrow=True,
                pyarrow_options={"partition_cols": partition_cols}
            )

    return True


def get_model_from_s3(s3_path: str):
    with conf_s3.fs.open(s3_path, 'rb') as f:
        model = joblib.load(f)
    return model


def write_model_to_s3(model, s3_path: str):
    with conf_s3.fs.open(s3_path, 'wb') as f:
        joblib.dump(model, f)
    return True
