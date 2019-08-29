# Author: Richard Correro

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class DataFramer(BaseEstimator, TransformerMixin):
    '''Class to convert array-like to Pandas Dataframe object.
    '''
    def __init__(self, columns=None):
        '''
        Parameters
        ----------
        columns : list of strings
            list of column names to be 
            to be applied to dataframe.
        '''
        
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''Converts array-like to Pandas DataFrame.

        Parameters
        ----------
        X : array-like
        
        Returns
        -------
        Pandas DataFrame object
        '''

        X = pd.DataFrame(X)
        
        # Apply column names
        if self.columns is not None:
            X.columns = self.columns
        
        return X
