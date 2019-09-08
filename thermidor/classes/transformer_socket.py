# Author: Richard Correro

from sklearn.base import TransformerMixin
from .estimator_socket import EstimatorSocket

class TransformerSocket(EstimatorSocket, TransformerMixin):
    '''A class which allows for treating transformers as
    model parameters.

    Parameters
    ----------
     estimator : Sci-kit learn estimator object
        If transformer is None or 'passthrough' then transform returns X.
    '''

    def transform(self, X):
        '''Method to transform X using the specified transformer.
        
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.
        '''
        
        return self.estimator.transform(X)
            
