"""
Wrapper per pyGAM per renderlo compatibile con sklearn Pipeline
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from pygam import LinearGAM, s, te, f, l


class PyGAMWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper per pyGAM LinearGAM compatibile con sklearn.
    
    Ottimizzato per interpolazione spaziale con coordinate (x, y).
    
    Parameters
    ----------
    n_splines : int, default=25
        Numero di splines per dimensione nel tensor product.
        Più alto = più flessibile ma rischio overfitting.
        Range tipico: 10-50
    
    lam : float, default=0.6
        Lambda - parametro di regolarizzazione.
        Più alto = più smooth (meno overfitting)
        Più basso = più flessibile (possibile overfitting)
        Range tipico: 0.001 - 100
    
    max_iter : int, default=100
        Numero massimo di iterazioni per l'ottimizzazione
    
    tol : float, default=1e-4
        Tolleranza per la convergenza
    
    use_tensor_product : bool, default=True
        Se True, usa tensor product te(0,1) per smooth bivariato su (x,y)
        Se False, usa somma di smooth univariati s(0) + s(1)
        Tensor product è migliore per dati spaziali!
    
    spline_order : int, default=3
        Ordine delle splines (grado del polinomio)
        2 = quadratico, 3 = cubico (default), 4 = quartico
    
    penalties : str, default='auto'
        Tipo di penalità: 'auto', 'derivative', o 'l2'
    
    fit_intercept : bool, default=True
        Se True, fitta un intercetta
    
    verbose : bool, default=False
        Se True, stampa info durante il fitting
    """
    
    def __init__(self,
                 n_splines=25,
                 lam=0.6,
                 max_iter=100,
                 tol=1e-4,
                 use_tensor_product=True,
                 spline_order=3,
                 penalties='auto',
                 fit_intercept=True,
                 verbose=False):
        
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.use_tensor_product = use_tensor_product
        self.spline_order = spline_order
        self.penalties = penalties
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.gam_model_ = None
        
    def fit(self, X, y):
        """
        Fit del modello GAM.
        
        Parameters
        ----------
        X : array-like o DataFrame, shape (n_samples, n_features)
            Features di input. Per dati spaziali: colonne 0 e 1 sono coordinate (x, y)
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
        
        # Costruisci la formula GAM
        if self.use_tensor_product:
            # Tensor product per smooth bivariato su coordinate spaziali
            formula = te(0, 1, 
                         n_splines=self.n_splines,
                         spline_order=self.spline_order)
        else:
            # Somma di smooth univariati
            formula = s(0, n_splines=self.n_splines, spline_order=self.spline_order) + \
                     s(1, n_splines=self.n_splines, spline_order=self.spline_order)
        
        # Se ci sono più di 2 feature, aggiungi smooth anche per quelle
        if X.shape[1] > 2:
            for i in range(2, X.shape[1]):
                formula = formula + s(i, n_splines=self.n_splines, spline_order=self.spline_order)
        
        # Crea il modello GAM con lam nel costruttore
        self.gam_model_ = LinearGAM(
            formula,
            lam=self.lam,  # Lambda nel costruttore!
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            fit_intercept=self.fit_intercept
        )
        
        # Fitta il modello
        self.gam_model_.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Predizione usando GAM.
        
        Parameters
        ----------
        X : array-like o DataFrame, shape (n_samples, n_features)
            Features dove fare le predizioni
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Valori predetti
        """
        if self.gam_model_ is None:
            raise ValueError("Il modello deve essere fittato prima di fare predizioni!")
        
        # Converti in numpy se è un DataFrame pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Predizione
        y_pred = self.gam_model_.predict(X)
        
        return y_pred
    
    def get_params(self, deep=True):
        """Ottieni i parametri del modello (necessario per sklearn)"""
        return {
            'n_splines': self.n_splines,
            'lam': self.lam,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'use_tensor_product': self.use_tensor_product,
            'spline_order': self.spline_order,
            'penalties': self.penalties,
            'fit_intercept': self.fit_intercept,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Imposta i parametri del modello (necessario per sklearn)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_optimal_lambda(self):
        """
        Ritorna il lambda ottimale trovato dal modello.
        Utile se hai usato lam=None per gridsearch.
        """
        if self.gam_model_ is None:
            raise ValueError("Il modello deve essere fittato prima!")
        return self.gam_model_.lam