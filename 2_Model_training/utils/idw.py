
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class IDWRegressor(KNeighborsRegressor):

    def __init__(self, power=2, n_neighbors=15, leaf_size=40, algorithm='kd_tree', n_jobs=-1, **kwargs):
        
        self.power = power
        
        def _idw_weights(distances):
            d = np.maximum(distances, 1e-10)
            return 1.0 / (d ** self.power)
            
        super().__init__(
            n_neighbors=n_neighbors,
            weights=_idw_weights,  
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
            **kwargs
        )