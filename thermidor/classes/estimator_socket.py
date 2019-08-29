# Author: Richard Correro

from sklearn.base import BaseEstimator


class EstimatorSocket(BaseEstimator):
    '''Class which allows for treating estimator as
    model parameters.

    Parameters
    ----------
    estimator : object, default=None
    '''
    
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        '''Method to fit the specified estimator.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
            
        Returns
        -------
        self : object
            Returns self.
        '''

        self.estimator.fit(X, y, **kwargs)
        
        return self
