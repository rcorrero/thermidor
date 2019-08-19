'''
thermidor
=============
thermidor is a Python module that streamlines the machine learning model 
development process. thermidor contains several utility functions and 
classes for use in data analysis and processing.
'''

# Import objects - flat is better than nested
from .classes import ColumnSelector
from .classes import TypeSelector
from .classes import DataFramer
from .classes import DummyEstimator
from .classes import EstimatorSocket
from .classes import TransformerSocket
from .classes import ClustererSocket

from .functions import get_categories, correlation
from .functions import date_extractor
