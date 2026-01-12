import pandas as pd
from MIBooster import MIXGBooster

def main():
    # Load data
    path_ice = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/Dataset/Synthetic/iceDF_synthetic.csv"
    path_grid = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/Dataset/Synthetic/gridDF_synthetic.csv"
    
    iceDF = pd.read_csv(path_ice)
    gridDF = pd.read_csv(path_grid)

    # 2. Prepare Training Data
    # Mapping columns: lon->x, lat->y, h_te_interp->target
    X_train = iceDF[["lon", "lat"]].values
    y_train = iceDF["h_te_interp"].values
    locs_train = iceDF[["lon", "lat"]].values

    # 3. Setup and Train
    print("Training MI Booster...")
    model = MIXGBooster(k=35, verbosity=0)
    model.fit(X_train, y_train, locs_train)

    # 4. Predict on Grid
    X_grid = gridDF[["lon", "lat"]].values
    
    print("Predicting on grid...")
    y_pred = model.predict(X_grid)

    # Save output
    gridDF["prediction"] = y_pred
    output_path = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/output/mbi_output.csv"
    gridDF.to_csv(output_path, index=False)
    print(f"Done. Saved to: {output_path}")

if __name__ == "__main__":
    main()