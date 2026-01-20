import numpy as np
import polars as pl
import gstools as gs
import s3fs

# --- 1. Connection & Config ---
fs = s3fs.S3FileSystem(anon=False)
s3_path = 's3://projet-benchmark-spatial-interpolation/data/synthetic'

SMALL_DIM = 100
LARGE_DIM = 1000
SEED = 20170519

# Setup the spatial model
model = gs.Matern(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, seed=SEED)

# --- 2. Define Generation Logic ---


def upload_synthetic_to_s3():
    # We define a dictionary of our 4 scenarios
    scenarios = {
        "S-G-Sm": {"type": "grid", "dim": SMALL_DIM},
        "S-G-Lg": {"type": "grid", "dim": LARGE_DIM},
        "S-NG-Sm": {"type": "nogrid", "dim": SMALL_DIM},
        "S-NG-Lg": {"type": "nogrid", "dim": LARGE_DIM},
    }

    for name, config in scenarios.items():
        print(f"Generating {name}...")

        if config["type"] == "grid":
            # Grid Logic
            coords = np.arange(config["dim"])
            field = srf.structured([coords, coords])
            xx, yy = np.meshgrid(coords, coords, indexing='ij')
            df = pl.DataFrame({
                'x': xx.flatten(),
                'y': yy.flatten(),
                'val': field.flatten()
            })
        else:
            # Non-Grid Logic
            rng = np.random.default_rng(SEED)
            num_points = config["dim"]**2
            x_rand = rng.uniform(0, config["dim"], num_points)
            y_rand = rng.uniform(0, config["dim"], num_points)
            field = srf((x_rand, y_rand))
            df = pl.DataFrame({
                'x': x_rand,
                'y': y_rand,
                'val': field
            })

        # --- 3. Direct Write to S3 ---
        target_uri = f"{s3_path}/{name}.parquet"
        print(f"    Writing to {target_uri}")

        # Polars write_parquet supports s3fs handles
        with fs.open(target_uri, mode='wb') as f:
            df.write_parquet(f)

    print("All synthetic datasets uploaded to S3.")


# Run the process
if __name__ == "__main__":
    upload_synthetic_to_s3()
