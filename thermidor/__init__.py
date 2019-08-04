'''
thermidor
=============
thermidor is a Python module that streamlines the machine learning model 
development process. thermidor contains several utility functions and 
classes for use in data analysis and processing.
'''

# Import objects - flat is better than nested
from thermidor.classes.column_selector import ColumnSelector
from thermidor.classes.type_selector import TypeSelector
from thermidor.classes.data_framer import DataFramer
from thermidor.functions.helpers import get_categories, correlation
