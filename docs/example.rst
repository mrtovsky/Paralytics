========
Examples
========
Simple illustrative examples on how you can quickly start using the **Paralytics** in your projects.

Discretization
--------------
Let's suppose that we want to discretize continuous variables. With use of the
Discretization transformer we can do it right away, because the transformer will
recognize which variables are of the continuous type.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import paralytics as prl


    # Fix the seed for reproducibility.
    SEED = 42
    np.random.seed(SEED)

    # Create available categories for non-numeric variable.
    sexes = ['female', 'male', 'child']

    # Generate example DataFrame.
    X = pd.DataFrame({
        'NormalVariable': np.random.normal(loc=0, scale=10, size=100),
        'UniformVariable': np.random.uniform(low=0, high=100, size=100),
        'IntVariable': np.random.randint(low=0, high=100, size=100),
        'Sex': np.random.choice(sexes, 100, p=[.5, .3, .2])
    })

    # Generate response variable.
    y = np.random.randint(low=0, high=2, size=100)

    # Do discretization.
    discretizer = prl.Discretizer(max_bins=5)
    X_discretized = discretizer.fit_transform(X, y)

The `X_discretized` dataframe is already a fully discretized equivalent of the
input dataframe `X`. First five rows will look like presented below.

====  ================  =================  =============  ======
  ..  NormalVariable    UniformVariable    IntVariable    Sex
====  ================  =================  =============  ======
   0  (-3.886, inf]     (33.151, inf]      (63.5, inf]    child
   1  (-3.886, inf]     (-inf, 24.071]     (-inf, 28.0]   female
   2  (-3.886, inf]     (-inf, 24.071]     (28.0, 63.5]   female
   3  (-3.886, inf]     (33.151, inf]      (63.5, inf]    male
   4  (-3.886, inf]     (33.151, inf]      (-inf, 28.0]   male
====  ================  =================  =============  ======
