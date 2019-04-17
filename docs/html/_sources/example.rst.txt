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

The ``X_discretized`` dataframe is already a fully discretized equivalent of the
input dataframe ``X``. First five rows will look like presented below.

====  ================  =================  =============  ======
  ..  NormalVariable    UniformVariable    IntVariable    Sex
====  ================  =================  =============  ======
   0  (-3.886, inf]     (33.151, inf]      (63.5, inf]    child
   1  (-3.886, inf]     (-inf, 24.071]     (-inf, 28.0]   female
   2  (-3.886, inf]     (-inf, 24.071]     (28.0, 63.5]   female
   3  (-3.886, inf]     (33.151, inf]      (63.5, inf]    male
   4  (-3.886, inf]     (33.151, inf]      (-inf, 28.0]   male
====  ================  =================  =============  ======

Forcing Context Manager
-----------------------
Let's suppose that we want to create a class which methods will only be available if
the instance is created using the context manager. We will get this functionality in the following way.

.. code-block:: python

   from paralytics import force_context_manager


   @force_context_manager
   class ExamplePrinter(object):
       def __init__(self):
           print('You just initiated me.')

       def open(self):
           print('I am opened.')

       def __exit__(exc_type, exc_value, traceback):
           print('...and now I am closed.')


   if __name__ == '__main__':
       example = ExamplePrinter()
       try:
           # Access methods without prior instance creation using the context
           # manager results in raising an exception.
           example.open()
       except RuntimeError as e:
           print(
               'Error message without using the context manager: \n\n"{}"\n\n'
               'Now we will try to do the same but with use of the '
               '`with` statement:\n'
               .format(e)
           )
           # This time no exception occurs.
           with ExamplePrinter() as example:
               example.open()

Assuming the Python code above is saved into a file called ``example.py`` it can be run at the
command line with the result below:

.. code-block:: console

   $ python example.py

   Error message without using the context manager:

   "Object of the ExamplePrinter should only be initialized with the `with` statement.
   Otherwise, the ExamplePrinter methods will not be available."

   Now we will try to do the same but with use of the `with` statement:

   You just initialized me.
   I am opened.
   ...and now I am closed.

Main advantage of using the ``force_context_manager`` function as a decorator allows to enforce
good practices and call the closing method **__exit__** without worrying about remembering it.
