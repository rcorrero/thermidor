# Author: Richard Correro

from sklearn.base import ClusterMixin

from .transformer_socket import TransformerSocket

class ClustererSocket(TransformerSocket, ClusterMixin):
    '''Class which allows for treating clusterers as
    model parameters.

    Parameters
    ----------
    estimator : object, default=None
        If estimator is None or 'passthrough' then transform returns X.
    '''
    def predict(self, X, sample_weight=None):
        '''Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        sample_weight : array-like, shape (n_samples,), optional
           The weights for each observation in X. If None, all 
           observations are assigned equal weight (default: None)
    
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        '''

        return self.estimator.predict(X, sample_weight)
    
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

    def score(self, X, y=None, sample_weight=None):
        '''Returns estimator's score method, if applicable.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        '''

        return self.estimator.score(X, y, sample_weight)
