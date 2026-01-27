import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def calc_w_sparse_fast(data, k=5):
    """
    Sparse Matrix
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nbrs.fit(data)
    _, indices = nbrs.kneighbors(data)
    
    n_samples = data.shape[0]
    weight_value = 1.0 / k
    
    #  CSR Matrix construction
    rows = np.repeat(np.arange(n_samples), k)
    cols = indices.flatten()
    weights = np.full(rows.shape, weight_value, dtype=np.float32)
    
    w_sparse = csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))
    return w_sparse

def mse_moranI_loss_optimized(y_true, y_pred, w_sparse, lamb):
    
    resid = y_pred - y_true
    lag_resid = w_sparse.dot(resid)
    grad = 2.0 * (resid + lamb * lag_resid)
    h_val = 2.0 * (1.0 + lamb)
    hess = np.full(y_true.shape, h_val, dtype=np.float64)
    
    return grad, hess


class MIXGBooster(xgb.XGBRegressor):
    def __init__(self, k=5, lamb=0.01, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.lamb = lamb
        self.w_sparse = None 
        
    def fit(self, X, y, locs=None):
        locations = locs if locs is not None else X
        
        if self.w_sparse is None:
            print(f"  [MI-GBT] Compute W (k={self.k})...", end=" ", flush=True)
            self.w_sparse = calc_w_sparse_fast(locations, k=self.k)
            
            
        def objective_func(y_true, y_pred):
            return mse_moranI_loss_optimized(y_true, y_pred, self.w_sparse, self.lamb)
            
        self.set_params(objective=objective_func)
        
        super().fit(X, y)
        return self


class SklearnMIXGBooster(BaseEstimator, RegressorMixin):
    def __init__(self, k=30, lamb=0.1, n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, **kwargs):
        self.k = k
        self.lamb = lamb
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model_ = None 

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else X
        if hasattr(X, "to_numpy"): X_arr = X.to_numpy()
        
        start_score = np.mean(y)
        
        self.model_ = MIXGBooster(
            k=self.k,
            lamb=self.lamb,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            verbosity=0,
            base_score=start_score, 
            **self.kwargs
        )
        
        self.model_.fit(X_arr, y, locs=X_arr)
        return self

    def predict(self, X):
        check_is_fitted(self, ['model_'])
        X_arr = X.values if hasattr(X, "values") else X
        if hasattr(X, "to_numpy"): X_arr = X.to_numpy()
        return self.model_.predict(X_arr)