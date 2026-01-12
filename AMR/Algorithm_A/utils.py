import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

epsilon = 1e-10

def local_moranI(y, w):
  y = y.reshape(-1)
  n = len(y)
  n_1 = n - 1
  z = y - y.mean()
  sy = max(y.std(), epsilon)
  z /= sy
  den = max((z * z).sum(), epsilon)
  zl = w * z
  mi = n_1 * z * zl / den
  scaler = MinMaxScaler()
  return scaler.fit_transform(mi.reshape(-1,1)).reshape(-1)


def row_standardize(w):
  new_data = []
  for i in range(0, w.shape[0]):
    wijs = w.getrow(i).data
    row_sum = sum(wijs)
    if row_sum == 0.0:
        print(("WARNING: ", i, " is an island (no neighbors)"))
    new_data.append([wij / row_sum for wij in wijs])
  w.data = np.array(new_data).flatten()
  return w

def calc_w(locations, k, weighted):
    mode = 'distance' if weighted else 'connectivity'
    w= kneighbors_graph(locations, n_neighbors=k, mode=mode, metric='haversine', include_self=False)
    # fill in epsilon value where data values is 0
    w.data = w.data + epsilon
    w.data = w.data**(-1)
    return row_standardize(w)

def smape(y_true, y_pred):
    den = np.max([(np.abs(y_pred) + np.abs(y_true)),np.array([epsilon]*len(y_pred))], axis=0)
    return np.mean( 
            np.abs(y_pred - y_true) / 
            (den/2))