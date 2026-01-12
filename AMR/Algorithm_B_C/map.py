import pandas as pd
import rasterio
import matplotlib.pyplot as plt

CSV_FILE      = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/Dataset/Synthetic/iceDF_synthetic.csv"      
RF_TIF_FILE   = "/home/onyxia/work/benchmark_spatial_interpolation/rf_output_randfor.tif"   
XGB_TIF_FILE  = "/home/onyxia/work/benchmark_spatial_interpolation/rf_output_xgboost.tif"    


def plot_training_points(csv_file: str):
    df = pd.read_csv(csv_file)

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
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )
    return data, extent


def plot_prediction(tif_file: str, title: str):
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


def plot_side_by_side(csv_file: str, rf_tif: str, xgb_tif: str):
    df = pd.read_csv(csv_file)
    if not {"lat", "lon", "h_te_interp"}.issubset(df.columns):
        raise ValueError("CSV must have columns: lat, lon, h_te_interp")

    rf_data, rf_extent   = read_tif(rf_tif)
    xgb_data, xgb_extent = read_tif(xgb_tif)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Training data
    sc = axs[0].scatter(
        df["lon"],
        df["lat"],
        c=df["h_te_interp"],
        cmap="terrain",
        s=8,
    )
    axs[0].set_title("Original training data")
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    plt.colorbar(sc, ax=axs[0], label="Elevation")

    # RF prediction
    im_rf = axs[1].imshow(
        rf_data,
        cmap="terrain",
        extent=rf_extent,
        origin="upper",
    )
    axs[1].set_title("Random Forest prediction")
    axs[1].set_xlabel("Longitude")
    axs[1].set_ylabel("Latitude")
    plt.colorbar(im_rf, ax=axs[1], label="Predicted elevation (RF)")

    # XGB prediction
    im_xgb = axs[2].imshow(
        xgb_data,
        cmap="terrain",
        extent=xgb_extent,
        origin="upper",
    )
    axs[2].set_title("XGBoost prediction")
    axs[2].set_xlabel("Longitude")
    axs[2].set_ylabel("Latitude")
    plt.colorbar(im_xgb, ax=axs[2], label="Predicted elevation (XGB)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("➡️ Mostro i punti originali...")
    plot_training_points(CSV_FILE)

    print("➡️ Mostro la mappa predetta (Random Forest)...")
    plot_prediction(RF_TIF_FILE, "Random Forest prediction (raster)")

    print("➡️ Mostro la mappa predetta (XGBoost)...")
    plot_prediction(XGB_TIF_FILE, "XGBoost prediction (raster)")

    print("➡️ Mostro confronto completo (training + RF + XGB)...")
    plot_side_by_side(CSV_FILE, RF_TIF_FILE, XGB_TIF_FILE)
