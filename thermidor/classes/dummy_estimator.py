# Author: Richard Correro

import warnings
from sklearn.base import BaseEstimator


class DummyEstimator(BaseEstimator):
    '''Class that allows for optimization
       over different estimators. 

    This class is deprecated. Use `EstimatorSocket`
    instead.
    '''

    def __init__(self,  estimator=None):
        '''
        Parameters
        ----------
        estimator : estimator object
        '''
        self.estimator = estimator

        warnings.warn(
            'DummyEstimator is deprecated, use EstimatorSocket instead.',
            DeprecationWarning
            )

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
