# Author: Richard Correro

from sklearn.base import ClusterMixin

from .transformer_socket import TransformerSocket

class ClustererSocket(TransformerSocket, ClusterMixin):
    '''A class which allows for treating clusterers as
    model parameters.

    Parameters
    ----------
    estimator : object, default=None
        If estimator is None or 'passthrough' then transform returns X.
    '''
    
    def fit_predict(self, X, y=None):
        '''Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
    
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        '''

        return self.estimator.fit_predict(X, y)
