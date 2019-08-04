# Author: Richard Correro

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class DataFramer(BaseEstimator, TransformerMixin):
    '''
    Input: columns - list of column names to be 
                     to be applied to dataframe
    Returns: X     - A pandas dataframe
    '''
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = pd.DataFrame(X)
        
        # Apply column names
        if self.columns is not None:
            X.columns = self.columns
        
        return X
