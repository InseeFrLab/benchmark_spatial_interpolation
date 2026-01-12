from interpolation import Interp
import rasterio
import pandas as pd
import json
import pyproj
import joblib

# Xarray
import xarray as xr
import rioxarray


# Scipy Spatial
from scipy.spatial import distance_matrix, KDTree, distance
from sklearn.neighbors import NearestNeighbors

# SK-learn functions
# from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Random Forest Regressors
from sklearn.ensemble import RandomForestRegressor  # SKLearn
from skranger.ensemble import RangerForestRegressor  # Ranger
import xgboost as xgb  # XGBoost

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.metrics import mean_squared_error, r2_score, max_error, accuracy_score
from sklearn.inspection import permutation_importance

import os, sys


def get_features_path(directory):
    tif_files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(directory, filename)
            key = os.path.splitext(filename)[0]  # Extract filename without extension
            tif_files[key] = file_path
    return tif_files


def sampleFromTIF(feat_labl, sample_from, sample_pts_df):
    """_summary_

    Args:
        feat_labl (_type_): _description_
        sample_from (_type_): _description_
        sample_pts_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    t = rasterio.open(sample_from, driver="GTiff")
    all_pt_coords = [
        (lon, lat) for lon, lat in zip(sample_pts_df["lon"], sample_pts_df["lat"])
    ]
    sampled_value = [sample[0] for sample in t.sample(all_pt_coords)]
    sampled_df = pd.DataFrame(all_pt_coords, columns=["lon", "lat"])
    sampled_df[feat_labl] = sampled_value
    return sampled_df


def oneHotEncoding(feat_df, feat_labl):
    """_summary_

    Args:
        feat_labl (_type_): _description_
        feat_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    encoder = OneHotEncoder()
    encoder.fit(feat_df[[feat_labl]])
    encoded_labl = encoder.transform(feat_df[[feat_labl]]).toarray()
    encoded_feat_df = pd.DataFrame(
        encoded_labl, columns=encoder.get_feature_names_out()
    )
    return encoded_feat_df


def normaliseScaling(feat_df, feat_labl):
    """_summary_

    Args:
        feat_labl (_type_): _description_
        feat_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    scaler = MinMaxScaler()
    scaled_feat_df = feat_df.copy()
    scaled_feat_num = scaled_feat_df[feat_labl].values.reshape(-1, 1)

    # Fit and transform the selected columns with the StandardScaler
    scaled_feat_df[feat_labl] = scaler.fit_transform(scaled_feat_num)
    scaled_features = scaled_feat_df.drop(columns=["lat", "lon"]).fillna(0)
    scaled_features.fillna(0)
    return scaled_features


class Distances:
    def __init__(self, icepts, gridpts) -> None:
        self.icepts = icepts
        self.gridpts = gridpts

    def convert_coordinates(lon, lat, from_epsg=4326, to_epsg=None):
        from_crs = pyproj.CRS.from_epsg(from_epsg)
        if to_epsg is None:
            # If no "to_epsg" is provided, it will convert to the default CRS of EPSG:3857 (Web Mercator)
            to_epsg = 3857
        to_crs = pyproj.CRS.from_epsg(to_epsg)
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        # Transform the coordinates
        x, y = transformer.transform(lon, lat)
        return x, y

    def dist_to(self, pts="icepts"):
        knn = NearestNeighbors(n_neighbors=100, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
        else:
            print("what points??")
        dist_df = pd.DataFrame(d)
        dist_df.columns = dist_df.columns.astype(str)
        return dist_df


class Regression:
    def __init__(self, pts_dataframe, pts_grid) -> None:
        self.df = pts_dataframe
        self.grid_raster = pts_grid
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def test_train(self):
        x = self.df.drop(columns=["h_te_interp"])
        y = self.df["h_te_interp"]  # Target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # VarianceThreshold from sklearn provides a simple baseline approach to feature selection
        return self.X_train, self.X_test, self.y_train, self.y_test

    # def sklearn_RFregression(self):
    #     self.test_train()

    #     # Random Forest Regressor
    #     sklearn_rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    #     sklearn_rf_model.fit(self.X_train, self.y_train)
    #     self.rf_evaluation(sklearn_rf_model)
    #     self.rf_model = sklearn_rf_model

    #     # -------- Grid Search CV --------
    #     make a dictionary of hyperparameters
    #     param_grid = {
    #         'n_estimators': [100, 200, 300],
    #         'max_depth': [None, 10, 20, 30],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4]
    #         }
    #     grid_search = GridSearchCV(estimator=sklearn_rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
    #     Fitting the Random Forest Regression model to the data
    #     grid_search.fit(self.X_train, self.y_train)
    #     best_rf_regressor = grid_search.best_estimator_
    #     best_params = grid_search.random_state = best_params_
    #     print(best_params)

    def sklearn_RFregression(self, use_model=None, params=None):
        self.test_train()
        # ----- Random Forest Regressor -----
        # use model (if any) && use params (if any)
        if use_model == None:
            sklearn_rf_model = RandomForestRegressor(**params)
        else:
            sklearn_rf_model = joblib.load(use_model)

        sklearn_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(sklearn_rf_model)
        self.rf_model = sklearn_rf_model
        return sklearn_rf_model

        # # -------- Grid Search CV --------
        # search_space = {
        #     "n_estimators" : [100, 200, 500],
        #     "max_depth" : [10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     "n_jobfs": [-1]
        #     }

        # # make a GridSearchCV object
        # GS = GridSearchCV(estimator = sklearn_rf_model,
        #                   param_grid = search_space,
        #                   # sklearn.metrics.SCORERS.keys()
        #                   scoring = ["r2", "neg_root_mean_squared_error"],
        #                   refit = "r2",
        #                   cv = 5,
        #                   verbose = 4)
        # GS.fit(self.X_train, self.y_train)
        # print(GS.best_estimator_)
        # print(GS.best_params_)
        # print(GS.best_score_)

    def ranger_RFregression(self):
        self.test_train()
        # ----- Ranger Regressor -----
        ranger_rf_model = RangerForestRegressor(n_jobs=-1)
        # Fitting the Random Forest Regression model to the data
        ranger_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(ranger_rf_model)
        self.rf_model = ranger_rf_model
        return ranger_rf_model

    def xgboost_RFregression(self):
        self.test_train()
        # ----- XGBoost Regressor -----
        xgb_rf_model = xgb.XGBRegressor(n_jobs=-1)

        # [XGB] Fitting the RF Regression model to the data
        xgb_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(xgb_rf_model)
        self.rf_model = xgb_rf_model
        return xgb_rf_model

    def rf_evaluation(self, rf_model):
        # Use the model to make predictions on the testing data
        y_pred = rf_model.predict(self.X_test)
        # Evaluate the performance of the model
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        max_err = max_error(self.y_test, y_pred)

        print("--------[RF Tests]--------")
        print("MSE:", mse)
        print("R-squared:", r2)
        print("Max error:", max_err)
        print("--------------------------")

        # if val_method == 'kfold'

        # print(rf_model.feature_importances_)
        # print(rf_model.n_features_in_)
        # print(rf_model.feature_names_in_)

    def save_rfmodel(self, rf_modelname):
        # save the model to disk
        joblib.dump(self.rf_model, rf_modelname)

    def output_tif(self, outname):
        pred_h = pd.DataFrame(self.grid_raster, columns=self.X_train.columns)
        pred_h.dropna(axis=1, inplace=True)

        rf_pred_target = self.rf_model.predict(pred_h)
        pred_h["pred_h"] = rf_pred_target
        latlon_h = pred_h[["lat", "lon", "pred_h"]]

        # export to Gtiff with 'lat' 'lon' and 'predicted h'
        xr_pred_h = xr.Dataset.from_dataframe(latlon_h.set_index(["lat", "lon"]))
        xr_pred_h.rio.set_crs("EPSG:4326")
        # xr_pred_h.rio.write_nodata(-9999)
        xr_pred_h.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        xr_pred_h.rio.to_raster(outname, driver="GTiff")

    # def cross_validation(self, rf_model, method='kfold'):
    #     if method=='kfold':
    #         scores=[]
    #         kFold=KFold(n_splits=10,random_state=42,shuffle=False)
    #         for train_index,test_index in kFold.split(X):
    #             print("Train Index: ", train_index, "\n")
    #             print("Test Index: ", test_index)

    #             X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #             knn.fit(X_train, y_train)
    #             scores.append(knn.score(X_test, y_test))
    #             knn.fit(X_train, y_train)
    #             scores.append(knn.score(X_test,y_test))
    #             print(np.mean(scores))
    #             0.9393939393939394
    #             cross_val_score(knn, X, y, cv=10)


def dist_to_pts(icepts, gridpts):
    d = Distances(icepts, gridpts)
    if gridpts is icepts:
        return d.dist_to("icepts")
    else:
        return d.dist_to("gridpts")


def regression(
    iceDF,
    gridDF,
    mode="sklearn",
    outname="outname.tif",
    save_rf_model=False,
    *params,
    **kwargs
):
    save_modelname = outname.removesuffix(".tif") + "_model.joblib"
    r = Regression(iceDF, gridDF)
    if mode == "sklearn":
        r.sklearn_RFregression(params=params)
        # if save_model
        if save_rf_model == True:
            r.save_rfmodel(save_modelname)

    elif mode == "ranger":
        r.ranger_RFregression()
        if save_rf_model == True:
            r.save_rfmodel(save_modelname)

    elif mode == "xgboost":
        r.xgboost_RFregression()
        if save_rf_model == True:
            r.save_rfmodel(save_modelname)

    r.output_tif(outname)


def use_rf_model(iceDF, gridDF, use_model=None, outname="outname.tif"):
    r = Regression(iceDF, gridDF)
    r.sklearn_RFregression(use_model=use_model)
    r.output_tif(outname)

    # r.rf_evaluation(Kfold)


def _test():
    pass


if __name__ == "__main__":
    _test()
