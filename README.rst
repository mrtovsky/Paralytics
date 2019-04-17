==========
Paralytics
==========

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: http://badge.fury.io/py/Paralytics

* Documentation: https://mrtovsky.github.io/Paralytics/.
* Repository: https://github.com/mrtovsky/Paralytics/.

What is it?
-----------
**Paralytics** package was created in order to simplify and accelerate repetitive
tasks during modeling and predictive analysis. It especially puts stronger emphasis
on data preprocessing, which is often the most arduous stage of modeling.

The purpose of this package is to reduce to a minimum time allocated on repetitive
activities preceding the problem-specific approach to a given problem, containing
among others optimization of the applied machine learning techniques, which is the
part that most of Data Scientists would like to devote the most energy to, however,
by poorly prepared data, it is often only a fraction of the total work time devoted
to the project.

Main Features
-------------
Highlighting the main functionalities of the **Paralytics**:

* Expanded **target encoding** of categorical variables using double cross-validation
  technique with additional regularisation preventing favoritism of sparse categories
  with reduction of excessive adjustment to the training set, effectively reducing
  overfitting.

* **Discretization** of continuous variables to ordinal using shallow decision tree or
  method based on Spearman's rank-order correlation.

* Processing data read into the
  `DataFrames <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_,
  including:

  * automatic unification of variable types,
  * grouping of sparse categories,
  * projecting of text variables whose unique elements symbolize a binary response onto
    binary variables,
  * imputation of missing data.

* Collinearity reduction using such factors as: **variance inflation factor** (VIF) or correlation.

Installation
------------

Dependencies
~~~~~~~~~~~~
**Paralytics** package requirements are checked and, if needed, installed during the installation
process automatically. Mainly used packages across the **Paralytics** are:

* `scikit-learn <https://scikit-learn.org/stable/>`_ (>= 0.20.1)
* `NumPy <http://www.numpy.org/>`_ (>= 1.15.4)
* `Pandas <https://pandas.pydata.org/>`_ (>= 0.23.4)

For visualizations:

* `seaborn <https://seaborn.pydata.org/>`_ (>= 0.9.0)
* `matplotlib <https://matplotlib.org/>`_ (>= 3.0.2)

The easiest way to install the package is using ``pip``: ::

    pip install paralytics

or directly from the github `repository <https://github.com/mrtovsky/Paralytics.git>`_: ::

    pip install git+https://github.com/mrtovsky/Paralytics.git

