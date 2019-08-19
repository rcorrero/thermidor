# Author: Richard Correro

import pandas as pd

def date_extractor(X, date_col='index',
                   start_date=None, end_date=None,
                   drop_nonfloat_cols=True,
                   drop_na_rows=True, drop_na_cols=True):
    '''Selects date ranges from arrays indexed by date and optionally
    drops na entries by row or column.
    
    Parameters
    ----------
    X : pandas dataframe
    
    date_col : str, default='index'
        Name of column containing dates. Dates must be strings
        of YYYYMMDD format.
    
    start_date : int, optional default=None
        Beginning of the range returned, inclusive (default is None, 
        in which case `date_selector` returns all days before `end_date`).
    
    end_date : int, optional default=None
        End of the range returned, exclusive (default is `None`,
        in which case `date_selector` returns all days after `start_date`).
        
    drop_nonfloat_cols : bool, optional default=True
        Whether to drop columns with any non-float64 entries.
        
    drop_na_rows : bool, optional default=True
        Whether to drop rows in which ALL entries are `na`.

    drop_na_cols : bool, optional default=True
    
    Returns
    ---------
    pandas dataframe
        Selected rows from `X`.
    '''
    if date_col == 'index':
        # Convert index to col for use in masks
        X['index'] = X.index
    
    
    # Select specified range
    if start_date == None and end_date == None:
        mask = (X[date_col] > 0)
        
    elif start_date == None:
        # end_date is specified
        
        mask = (X[date_col] < end_date)
        
    elif end_date == None:
        # start_date is specified
        
        mask = (X[date_col] >= start_date)
        
    else:
        # Both start_date and end_date specified
        
        mask = (X[date_col] >= start_date) & (X[date_col] < end_date)
        
    
    X_new = X.loc[mask]
    
    
    # Drop `index` column
    if date_col == 'index':
        X.drop('index', axis=1, inplace=True)
        X_new.drop('index', axis=1, inplace=True)
        
    
    if drop_nonfloat_cols:
       
        # Convert string columns to float, if possible
        X_new = X_new.apply(pd.to_numeric, errors='ignore')
        
        # Drop columns containing non-float entries
        X_new.drop([i for i in X_new.columns.values.tolist() 
                if X_new.dtypes[i] != 'float64'], axis=1, inplace=True)
        
        
    if drop_na_rows:
        # See if any rows are all na
        na_rows = X_new.index[X_new.isnull().all(axis=1)]
        
        X_new.drop(na_rows, axis=0, inplace=True)
    
    if drop_na_cols:
        # Remove any securities with missing returns
        na_cols = X_new.columns[X_new.isnull().any()]
        
        X_new.drop(na_cols, axis=1, inplace=True)
        
    
    return X_new
