History
=======

0.2.2 ()
------------------
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
