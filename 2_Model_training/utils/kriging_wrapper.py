"""
Wrapper per PyKrige per renderlo compatibile con sklearn Pipeline
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from pykrige.ok import OrdinaryKriging


class PyKrigeWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper per PyKrige OrdinaryKriging compatibile con sklearn.
    
    Parameters
    ----------
    variogram_model : str, default='spherical'
        Modello di variogramma da usare.
        Opzioni: 'linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect'
    
    nlags : int, default=6
        Numero di lag per il calcolo del variogramma
    
    weight : bool, default=False
        Se True, usa pesi per il variogramma
    
    anisotropy_scaling : float, default=1.0
        Fattore di scaling per anisotropia
    
    anisotropy_angle : float, default=0.0
        Angolo di anisotropia in gradi
    
    enable_plotting : bool, default=False
        Se True, abilita il plotting del variogramma
    
    verbose : bool, default=False
        Se True, stampa informazioni durante il fitting
    
    exact_values : bool, default=True
        Se True, forza i valori esatti ai punti di training
    
    pseudo_inv : bool, default=False
        Se True, usa pseudo-inversa invece di inversione diretta
    
    coordinates_type : str, default='euclidean'
        Tipo di coordinate: 'euclidean' o 'geographic'
    """
    
    def __init__(self, 
                 variogram_model='spherical',
                 nlags=6,
                 weight=False,
                 anisotropy_scaling=1.0,
                 anisotropy_angle=0.0,
                 enable_plotting=False,
                 verbose=False,
                 exact_values=True,
                 pseudo_inv=False,
                 coordinates_type='euclidean'):
        
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.weight = weight
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        self.enable_plotting = enable_plotting
        self.verbose = verbose
        self.exact_values = exact_values
        self.pseudo_inv = pseudo_inv
        self.coordinates_type = coordinates_type
        self.ok_model_ = None
        
    def fit(self, X, y):
        """
        Fit del modello di kriging.
        
        Parameters
        ----------
        X : array-like o DataFrame, shape (n_samples, 2)
            Coordinate (x, y) dei punti di training
        y : array-like, shape (n_samples,)
            Valori target
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Converti in numpy se è un DataFrame pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Estrai coordinate
        x_coords = X[:, 0]
        y_coords = X[:, 1]
        
        # Crea il modello OrdinaryKriging
        self.ok_model_ = OrdinaryKriging(
            x_coords,
            y_coords,
            y,
            variogram_model=self.variogram_model,
            nlags=self.nlags,
            weight=self.weight,
            anisotropy_scaling=self.anisotropy_scaling,
            anisotropy_angle=self.anisotropy_angle,
            enable_plotting=self.enable_plotting,
            verbose=self.verbose,
            exact_values=self.exact_values,
            pseudo_inv=self.pseudo_inv,
            coordinates_type=self.coordinates_type
        )
        
        return self
    
    def predict(self, X):
        """
        Predizione usando il kriging.
        
        Parameters
        ----------
        X : array-like o DataFrame, shape (n_samples, 2)
            Coordinate (x, y) dove fare le predizioni
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Valori predetti
        """
        if self.ok_model_ is None:
            raise ValueError("Il modello deve essere fittato prima di fare predizioni!")
        
        # Converti in numpy se è un DataFrame pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Estrai coordinate
        x_coords = X[:, 0]
        y_coords = X[:, 1]
        
        # Esegui la predizione
        y_pred, _ = self.ok_model_.execute('points', x_coords, y_coords)
        
        return y_pred
    
    def get_params(self, deep=True):
        """Ottieni i parametri del modello (necessario per sklearn)"""
        return {
            'variogram_model': self.variogram_model,
            'nlags': self.nlags,
            'weight': self.weight,
            'anisotropy_scaling': self.anisotropy_scaling,
            'anisotropy_angle': self.anisotropy_angle,
            'enable_plotting': self.enable_plotting,
            'verbose': self.verbose,
            'exact_values': self.exact_values,
            'pseudo_inv': self.pseudo_inv,
            'coordinates_type': self.coordinates_type
        }
    
    def set_params(self, **params):
        """Imposta i parametri del modello (necessario per sklearn)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self