thermidor &mdash; Richard Correro
==============================

thermidor is a Python module that streamlines the machine learning model development process. thermidor contains several utility functions and classes for use in data analysis and processing.

thermidor provides a framework for generalizing and extending machine learning models built in Sci-kit Learn. thermidor includes estimator, transformer and clusterer sockets which allow for treating steps in a Sci-kit Learn pipeline as parameters. These parameters may be passed to `RandomizedSearchCV` or `GridSearchCV` for use in cross-
validation.

Project Organization
------------
```
.
├── LICENSE
├── README.md
├── setup.py
└── thermidor
    ├── __init__.py
    ├── classes
    │   ├── __init__.py
    │   ├── clusterer_socket.py
    │   ├── column_selector.py
    │   ├── data_framer.py
    │   ├── dummy_estimator.py
    │   ├── estimator_socket.py
    │   ├── estimator_socket_cv.py
    │   ├── transformer_socket.py
    │   └── type_selector.py
    └── functions
        ├── __init__.py
        ├── date_extractor.py
        ├── grid_search.py
        └── helpers.py

```    

-------------
Sources:
-------------
[TypeSelector, ColumnSelector](https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/)

[correlation](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/House_Prices.ipynb)

----------
Name:
----------

This project was started in the month of [Thermidor](https://en.wikipedia.org/wiki/Thermidor), year CCXXVII.
------------
Created by Richard Correro in 2019. Contact me at rcorrero at stanford dot edu