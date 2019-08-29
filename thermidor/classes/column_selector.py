# From:
# https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    '''Class which allows for treating column selection as model parameter.
    
    Useful in situations in which separate columns in array contain data
    on which different transformations have been applied.
    '''
    
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''Selects and returns column specified in __init___
        
        Parameters:
        -----------
        X : array-like

        Returns
        -------
        Pandas series or list
        '''
        assert isinstance(X, pd.DataFrame), 'X is not a DataFrame.'

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
