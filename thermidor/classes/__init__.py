'''
thermidor/classes
==============
this module contains all utility classes in thermidor. 
These classes extend the functionality of Python data science tools.
'''

# Import objects
from .column_selector import ColumnSelector
from .type_selector import TypeSelector
from .data_framer import DataFramer
from .dummy_estimator import DummyEstimator
from .estimator_socket import EstimatorSocket
from .estimator_socket_cv import EstimatorSocketCV
from .transformer_socket import TransformerSocket
from .clusterer_socket import ClustererSocket
