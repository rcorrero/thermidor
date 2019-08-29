# Author: Richard Correro

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed


def grid_search(estimator, param_grid, score_fun, X, y=None, maximize=True,
                n_jobs=-1, verbose=False, print_params=False):
    '''Exhaustive search over specified parameter values for an estimator. 
    
    Useful in cases in which cross-validation is not suitable.


    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    score_fun : callable
        A callable which takes the following inputs:
            estimator : estimator object
            
            X : array-like

            **kwargs : optional

    X : array-like

    y : array-like, optional default=None

    maximize : boolean, optional default=True
        Whether to maximize or minimize score_fun.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose : bool, optional default=False

    print_params : bool, optional  default=False

    
    Returns
    -------
    (best_estimator, best_score, score_dict) : tuple
        best_estimator : estimator object
            Estimator that was chosen by the search, i.e. estimator 
            which gave highest score (or smallest loss if specified) 
            on the left out data.
        
        best_score : float or int
            Score of best_estimator.

        score_dict : dictionary
            Dictionary mapping estimator objects to their scores.
    '''
    grid = ParameterGrid(param_grid)
    
    if verbose != False:
        print('Fitting %i candidates (tasks)' % len(list(grid)))
        
    # Try all Cartesian products
    score_dict = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_grid_search)
        (estimator, g, score_fun, X) for g in grid
    )
    
    if maximize:
        best_score = max([elem[1] for elem in score_dict])  # maximum value
    else:
        best_score = min([elem[1] for elem in score_dict])  # minimum value
        
    best_estimator = [elem[0] for elem in score_dict if elem[1] == best_score][0]
    
    if print_params: 
        print ("OOB: %0.5f" % best_score)
        print ("Grid:", best_estimator.get_params())
    
    return (best_estimator, best_score, score_dict)



def _grid_search(estimator, g, score_fun, X):
    '''Scoring function for grid_search.
    
    Parameters
    ----------
    estimator : estimator object

    g : ParameterGrid object
    
    score_fun: Scoring function. See grid_search for details.
    
    X : array-like

    Returns
    -------
    (estimator, score) : tuple
        estimator: estimator object

        score : Score of estimator
    '''       
    estimator.set_params(**g)
    
    estimator.fit(X)
        
    score = score_fun(estimator, X)
        
    return (estimator, score)
