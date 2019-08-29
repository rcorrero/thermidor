# From:
# https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class TypeSelector(BaseEstimator, TransformerMixin):
    '''Class which allows for selecting subset of 
    Pandas DataFrame columns by type.
    '''
    def __init__(self, dtype):
        '''
        Parameters
        ----------
        dtype: type
        '''
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''Selects columns containing only type `dtype`.
        
        Parameters
        ----------
        X : array-like

        Returns
        -------
        Pandas DataFrame
        '''
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
