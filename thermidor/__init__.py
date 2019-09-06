'''
thermidor
=============
thermidor is a Python module that streamlines the machine learning model 
development process. thermidor contains several utility functions and 
classes for use in data analysis and processing.

thermidor provides a framework for generalizing and extending machine
learning models built in Sci-kit Learn. thermidor includes estimator, 
transformer and clusterer sockets which allow for treating steps in
a Sci-kit Learn pipeline as parameters. These parameters may be 
passed to `RandomizedSearchCV` or `GridSearchCV` for use in cross-
validation.
'''

# Import objects - flat is better than nested
from .classes import ColumnSelector
from .classes import TypeSelector
from .classes import DataFramer
from .classes import DummyEstimator
from .classes import EstimatorSocket
from .classes import EstimatorSocketCV
from .classes import TransformerSocket
from .classes import ClustererSocket

from .functions import get_categories, correlation
from .functions import date_extractor
from .functions import grid_search
