History
=======

X.Y.Z (DD.MM.YYYY)
------------------
* Created ``DevelopmentStageWarning``
    * Creates custom warning raised when the functionality is to some extent put into use,
      but not yet fully implemented.

0.3.2 (03.10.2019)
------------------
* Fixed ``Discretizer``
    * Properly addresses the cut-offs in the ``transform`` method, when not enough
      split points is provided.
    * Adds ``random_state`` parameter for reproducibility.

* Added ``fit_intercept`` parameter to the ``VIFSelector``
    * Extends functionality through the ability to control the model intercept.
      Statsmodels natively do not include the intercept in the linear regression
      so to properly execute the VIF selection it is recommended to add the intercept
      and aforementioned parameter provides this.

0.3.1 (09.07.2019)
------------------
* Fixed ``MANIFEST.in``
    * Includes ``extras_requirements.json`` to distribution package.

0.3 (09.07.2019)
----------------
* Initiated ``xai`` subpackage
    * Creates base class ``ExplainerMixin`` for all explainers with the **fit** & **explain** 
      convention (familiar from scikit-learn API).
    * Releases, in the development version, the ``xai.FeatureEffectExplainer`` implementing
      calculation of **Partial Dependence Plot**, **Individual Conditional Expectation** and
      **Marginal Plot**.
* Optimized ``Discretizer``
    * Limits the transformations only to continuous variables by not taking into consideration
      categorical variables and leaving them unchanged.
* Created optional dependencies
    * Reduces the number of dependent packages by creating optional functionalities 
      that require additional installation of the extra requirements.
* Renamed ``collinearity_reduction`` module
    * Changes the module name to ``feature_selection`` to prepare it for future
      expansion in this direction.
* Modified ``utils.is_numeric``
    * Excludes from checking for numericity categorical pandas.Series.
    * Adds the ability to disable the attempt to project on a numeric type.
* Moved ``mathy`` module
    * Changes the location of ``mathy`` module to ``utils`` subpackage.
      The new location is as follows: ``utils.mathy``.
* Fixed missing values temporary imputation inside the ``VIFSelector``
    * Imputs the missing values ​​before checking the NaN condition.

0.2.2 (19.06.2019)
------------------
* Added ``PandasFeatureUnion`` transformer
    * Creates ``feature_union`` module with ``PandasFeatureUnion`` transformer implemented
      that concatenates multiple transformers returning pandas.DataFrame.
* Fixed ``preprocessing.ColumnProjector`` corner case handling
    * Checks whether the list of columns to be projected is non-empty, because
      when projecting an empty list of columns onto the **category** dtype,
      **ValueError** was raised.
* Expanded ``TargetEncoding`` docstring and unified types
    * Adds notes on the use of the indicated encoder.
    * Projects to make the transformed dataframe's columns being outputted as floating
      point type.
* Changed ``utils.check_column_existance`` name
    * Repairs the typo in the function name by changing it to: ``utils.check_column_existence``.
* Expanded Sphinx documentation.
    * Adds example usage of ``force_context_manager`` function.
* Added ``utils.check_continuity`` function
    * Creates function that asserts whether the variable is truly continuous at a given
      repetition threshold.

0.2.1 (31.03.2019)
------------------
* Fixed references to correct redirections
    * Repairs links to valid websites.

0.2 (31.03.2019)
----------------
* Changed the topology of the repository
    * Transfers the ``column_parsing`` module to the preprocessing subpackage and
      changes the name to a more adequate in terms of actual functionality.
    * Creates ``preprocessing.imputation`` module used for imputation of missings.
* Added Sphinx documentation
    * Creates HTML documentation of the package with use of the Github Pages.
    * Standardizes the docstrings across the whole package.
    * Adds initial example of usage for the ``paralytics.Discretizer`` transformer.
* Created utilities for web scraping
    * New module ``preprocessing.scraping`` presenting the base class for scraping
      using the Selenium package.
* Fixed functionality of defective transformers
    * Transfers key operations of the ``preprocessing.ColumnProjector`` transformer
      to the **fit** method to avoid errors when applying to corner cases.
    * Adapts the functionality of the ``TargetEncoder`` to be able to apply its
      transformations to entire DataFrame without first choosing only variables of
      the type: **category**.
* Initialized XAI subpackage
    * Initializes directory for the Explainable Artificial Intelligence subpackage
      which will be developed over the 0.2.X versions.

0.1 (24.03.2019)
----------------
* Initial package
    * Initializes the **Paralytics** package.
