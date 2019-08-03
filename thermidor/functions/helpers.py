# Author: Richard Correro

import pandas as pd
import scipy.stats as stats


# Derive cats from full dataset
def get_categories(df, cols):
    '''
    Parameters
    
    df : Pandas dataframe
    
    cols : Columns to derive categories from
    
    Returns
    
    cats : Dictionary mapping col name to categories
    '''
    
    # Dict mapping colname to levels
    cats = {}
    
    for col in cols:
        cats[col] = df[col].unique()
    
    return cats



# Inspired by:
# https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/House_Prices.ipynb

def correlation(data, threshold = -1):
    '''Input: data - A pandas dataframe.
       threshold - absolute min correlation.
       Returns: a list of tuples.'''
    factors_paired = [(i,j) for i in data.columns for j in data.columns[data.columns.get_loc(i)+1:]]

    corr_list = []
    
    for k in factors_paired:
        col_1 = data[k[0]]
        col_2 = data[k[1]]
        
        if col_1.dtype != 'object' and col_2.dtype != 'object':
            corr = stats.pearsonr(col_1, col_2)[0]
                    
            if abs(corr) >= threshold:
                corr_list.append((corr, col_1.name, col_2.name))
        
        else:
            corr = stats.spearmanr(col_1,col_2)[0]
                    
            if abs(corr) >= threshold:
                corr_list.append((corr, col_1.name, col_2.name))
            
    return corr_list


