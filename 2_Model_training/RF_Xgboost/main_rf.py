import pandas as pd
from random_forest import Regression

def main():
    iceDF = pd.read_csv("/home/onyxia/work/benchmark_spatial_interpolation/2_Model_training/spaitalheight/iceDF_synthetic.csv")
    gridDF = pd.read_csv("/home/onyxia/work/benchmark_spatial_interpolation/2_Model_training/spaitalheight/gridDF_synthetic.csv")

    r = Regression(iceDF, gridDF)

    r.sklearn_RFregression(params={})

    r.output_tif("rf_output_randfor.tif")

if __name__ == "__main__":
    main()
