import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# File paths
CSV_FILE      = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/Dataset/Synthetic/iceDF_synthetic.csv"
MBI_CSV_FILE  = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/output/mbi_output.csv"
RF_TIF_FILE   = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/output/rf_output_randfor.tif"   
XGB_TIF_FILE  = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/output/rf_output_xgboost.tif"    


def plot_training_points(csv_file: str):
    """Plots the original training data points."""
    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    if not {"lat", "lon", "h_te_interp"}.issubset(df.columns):
        raise ValueError("CSV must have columns: lat, lon, h_te_interp")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df["lon"],
        df["lat"],
        c=df["h_te_interp"],
        cmap="terrain",
        s=10,
    )
    plt.colorbar(sc, label="Elevation (h_te_interp)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Original elevation points (training data)")
    plt.tight_layout()
    plt.show()


def read_tif(tif_file: str):
    """Helper function to read TIF raster data."""
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )
    return data, extent


def plot_raster_prediction(tif_file: str, title: str):
    """Plots a TIF prediction file."""
    data, extent = read_tif(tif_file)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        cmap="terrain",
        extent=extent,
        origin="upper",
    )
    plt.colorbar(im, label="Predicted elevation")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mbi_prediction(csv_file: str):
    """Plots the MBI prediction from CSV."""
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(8, 6))
    # We use scatter because MBI output is a CSV list of points, not a raster image
    sc = plt.scatter(
        df["lon"],
        df["lat"],
        c=df["prediction"],
        cmap="terrain",
        s=1, # Small size to look like a grid
        marker='s'
    )
    plt.colorbar(sc, label="Predicted elevation")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("MBI Prediction (CSV)")
    plt.tight_layout()
    plt.show()


def plot_comparison(train_csv: str, mbi_csv: str, rf_tif: str, xgb_tif: str):
    """Plots Training data, MBI, RF, and XGB side by side."""
    
    # Load Training Data
    df_train = pd.read_csv(train_csv)
    if not {"lat", "lon", "h_te_interp"}.issubset(df_train.columns):
        raise ValueError("Training CSV must have columns: lat, lon, h_te_interp")

    # Load MBI Data
    df_mbi = pd.read_csv(mbi_csv)
    
    # Load RF and XGB Data
    rf_data, rf_extent   = read_tif(rf_tif)
    xgb_data, xgb_extent = read_tif(xgb_tif)

    # Create 4 subplots side by side
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Training data
    sc1 = axs[0].scatter(
        df_train["lon"],
        df_train["lat"],
        c=df_train["h_te_interp"],
        cmap="terrain",
        s=8,
    )
    axs[0].set_title("1. Original Training Data")
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    plt.colorbar(sc1, ax=axs[0], label="Elevation")

    # 2. MBI prediction (Scatter visualization of the grid)
    sc2 = axs[1].scatter(
        df_mbi["lon"],
        df_mbi["lat"],
        c=df_mbi["prediction"],
        cmap="terrain",
        s=2, # Small marker to simulate raster
        marker='s'
    )
    axs[1].set_title("2. MBI Prediction")
    axs[1].set_xlabel("Longitude")
    axs[1].set_ylabel("Latitude")
    plt.colorbar(sc2, ax=axs[1], label="Prediction")

    # 3. RF prediction (Raster)
    im_rf = axs[2].imshow(
        rf_data,
        cmap="terrain",
        extent=rf_extent,
        origin="upper",
    )
    axs[2].set_title("3. Random Forest Prediction")
    axs[2].set_xlabel("Longitude")
    axs[2].set_ylabel("Latitude")
    plt.colorbar(im_rf, ax=axs[2], label="Prediction")

    # 4. XGB prediction (Raster)
    im_xgb = axs[3].imshow(
        xgb_data,
        cmap="terrain",
        extent=xgb_extent,
        origin="upper",
    )
    axs[3].set_title("4. XGBoost Prediction")
    axs[3].set_xlabel("Longitude")
    axs[3].set_ylabel("Latitude")
    plt.colorbar(im_xgb, ax=axs[3], label="Prediction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("➡️ Showing original points...")
    plot_training_points(CSV_FILE)

    print("➡️ Showing MBI prediction (CSV)...")
    plot_mbi_prediction(MBI_CSV_FILE)

    print("➡️ Showing RF prediction (Raster)...")
    plot_raster_prediction(RF_TIF_FILE, "Random Forest prediction (raster)")

    print("➡️ Showing XGB prediction (Raster)...")
    plot_raster_prediction(XGB_TIF_FILE, "XGBoost prediction (raster)")

    print("➡️ Showing complete comparison (Train -> MBI -> RF -> XGB)...")
    plot_comparison(CSV_FILE, MBI_CSV_FILE, RF_TIF_FILE, XGB_TIF_FILE)