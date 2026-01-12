import pandas as pd
from random_forest import Regression


def main():
    iceDF_file = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/Dataset/Synthetic/iceDF_synthetic.csv"
    gridDF_file = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/Dataset/Synthetic/gridDF_synthetic.csv"
    output_tif = "/home/onyxia/work/benchmark_spatial_interpolation/AMR/output/rf_output_xgboost.tif"

    iceDF = pd.read_csv(iceDF_file)
    gridDF = pd.read_csv(gridDF_file)

    r = Regression(iceDF, gridDF)

    r.xgboost_RFregression()

    r.output_tif(output_tif)


if __name__ == "__main__":
    main()
