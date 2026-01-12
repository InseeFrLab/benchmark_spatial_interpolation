import lightgbm
import xgboost

import numpy as np
from utils import *

epsilon = 1e-10


def pseudohuber_moranI_loss(y_pred, y_true, w):
  i_pred = local_moranI(y_pred, w)
  i_true = local_moranI(y_true, w)
  d = (i_true-i_pred)
  delta = 1  
  scale = 1 + (d / delta) ** 2
  scale_sqrt = np.sqrt(scale)
  grad = d / scale_sqrt 
  hess = (1 / scale) / scale_sqrt
  return grad, hess

def mse_moranI_loss(y_pred, y_true, w):
  i_pred = local_moranI(y_pred, w)
  i_true = local_moranI(y_true, w)
  grad = 2*(i_true-i_pred)
  hess = np.repeat(2,i_true.shape[0])
  return grad, hess

def mse_loss(y_pred, y_true):
  grad = 2*(y_true-y_pred)
  hess = np.repeat(2,y_true.shape[0])
  return grad, hess  

def weighted_loss1(y_pred, y_true, w, l=0.01):
  grad_mi, hess_mi = pseudohuber_moranI_loss(y_pred, y_true, w)
  grad_mse, hess_mse = mse_loss(y_pred, y_true)
  return grad_mse + l*grad_mi, hess_mse + l*hess_mi

def weighted_loss2(y_pred, y_true, w, l=0.01):
  grad_mi, hess_mi = mse_moranI_loss(y_pred, y_true, w)
  grad_mse, hess_mse = mse_loss(y_pred, y_true)
  return grad_mse + l*grad_mi, hess_mse + l*hess_mi



class MILGBooster(lightgbm.LGBMRegressor):
    def __init__(self, k=5, weighted=True, lamb= 0.01, loss_type='mse',
        boosting_type='gbdt', num_leaves=31, max_depth=-1, 
        learning_rate=0.1, n_estimators=100, 
        subsample_for_bin=200000, objective=None, class_weight=None, 
        min_split_gain=0.0, min_child_weight=0.001, 
        min_child_samples=20, subsample=1.0, subsample_freq=0, 
        colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, 
        random_state=None, n_jobs=None, importance_type='split', 
        **kwargs):
        super().__init__(boosting_type=boosting_type, num_leaves=num_leaves,
        max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
        subsample_for_bin=subsample_for_bin, objective=objective, class_weight=class_weight,
        min_split_gain=min_split_gain, min_child_weight=min_child_weight,
        min_child_samples=min_child_samples, subsample=subsample, subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        random_state=random_state, n_jobs=n_jobs, importance_type=importance_type,
         **kwargs)
        self.k = k
        self.lamb = lamb
        self.loss_type=loss_type
        self.w = None
        self.weighted = weighted
        

    def weighted_loss(self, y_pred, y_true):
        if self.loss_type == 'mse':
            return weighted_loss2(y_pred, y_true, w=self.w, l=self.lamb)
        else:
            return weighted_loss1(y_pred, y_true, w=self.w, l=self.lamb)

    def set_objective(self):
        self.set_params(objective=self.weighted_loss)


    def fit(self, X, y, locs=None):
        locations = locs if locs is not None else np.deg2rad(X[:,[-1,-2]])
        self.w = calc_w(locations, k=self.k, weighted=self.weighted) if self.w is None else self.w
        self.set_objective()
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

class MIXGBooster(xgboost.XGBRegressor):
    def __init__(self, k=5, weighted=True, lamb= 0.01, loss_type='mse',
        **kwargs):
        super().__init__( **kwargs)
        self.k = k
        self.lamb = lamb
        self.loss_type=loss_type
        self.w = None
        self.weighted = weighted
        

    def weighted_loss(self, y_pred, y_true):
        if self.loss_type == 'mse':
            return weighted_loss2(y_pred, y_true, w=self.w, l=self.lamb)
        else:
            return weighted_loss1(y_pred, y_true, w=self.w, l=self.lamb)

    def set_objective(self):
        self.set_params(objective=self.weighted_loss)

    def fit(self, X, y, locs=None):
        locations = locs if locs is not None else np.deg2rad(X[:,[-1,-2]])
        self.w = calc_w(locations, k=self.k, weighted=self.weighted) if self.w is None else self.w
        self.set_objective()
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)
