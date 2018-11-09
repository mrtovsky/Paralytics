import numpy as np
import pandas as pd
import scipy.stats as stats
import inspect
import pandas.core.algorithms as algos

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from .utils import Utils


class Discretization(BaseEstimator, TransformerMixin, Utils):
    """Discretizes variables in a given data set.
    
    Reduces continuous variables to a finite number of intervals with use of 
    declared methods.
    
    Parameters
    ----------
    method: string {'sapling', 'spearman'}
        sapling:
            Submethod based on the DecisionTreeClassifier.
        spearman:
            Submethod based on the Spearman's rang correlation. Divides the
            values into subsequent quartiles as long as it doesn't get full
            monotonicity. If this doesn't happen, it divides values with use 
            of quantiles into the declared minimum number of buckets.
            Using this method with parameter formula set to 'median' may throw 
            RuntimeWarning for fixed vector values in one of the input vectors
            becuase there is no point in tracking mutual change of two vectors
            when one vector doesn't change. In this case you can run your code
            with use of this command: python -W ignore foo.py 

    formula: string {'mean', 'median'} (default: 'mean')
        Chooses the formula that representatives will be chosen to check the 
        Spearman's rank correlation.

    max_bins: int (default: 20)
        Maximum number of bins that will be created.

    min_bins: int (default: 3)
        Minimum number of bins that will be created.

    max_tree_depth: int
        Specifies maximum tree depth.

    min_samples_leaf: float (default: .05)
        Specifies the minimum part of the entire population that must be 
        included in the leaf.

    Attributes
    ----------
    bins_: dictionary, length = n_features
        Dictionary of upper limits of successive intervals excluding the 
        maximum value which length equals the number of features in the data
        passed.

    """

    def __init__(self, method='spearman', formula='mean', 
                 max_bins=20, min_bins=3, max_tree_depth=None,
                 min_samples_leaf=.05):
        icf = inspect.currentframe()    
        args, _, _, values = inspect.getargvalues(icf)
        values.pop('self')

        for param, value in values.items():
            setattr(self, param, value)

    def fit(self, X, y=None, **params):
        """Fit the binning with X by extracting upper limits of right-closed
        intervals. 

        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            Training data of independent variable values.

        y: array-like, shape (n_samples, )
            Vector of target variable values corresponding to X data.

        Returns
        -------
        self: object
            Returns the instance itself.
        
        """
        assert isinstance(X, pd.DataFrame), 'Input must be an instance of \
                                             pandas.DataFrame()'

        call_method = getattr(self, self.method)
        self.bins_ = {}
        for col in X.columns.values:
            # Checking whether columns are non-binary numeric (excluding nans) 
            if self.is_numeric(X[col]):
                self.bins_[col] = call_method(X[col], np.asarray(y).ravel(), 
                                              **params)
            else:
                self.bins_[col] = np.unique(X[col].astype(str))

        return self

    def transform(self, X):
        """Apply discretization on X.

        X is projected on the bins previously extracted from a training set.
        
        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            New data with n_samples as its number of samples.
        
        Returns
        -------
        X_new: DataFrame, shape (n_samples_new, n_features)
            X data with substituted values to their respective labels being
            string type. 
        
        """
        # Checking whether estimator fitting was done.
        try:
            getattr(self, 'bins_')
        except AttributeError:
            raise RuntimeError('Could not find the attribute.\n'
                               + 'Fitting is necessary before you do '
                               + 'the transformation.')

        assert isinstance(X, pd.DataFrame), 'Input must be an instance of \
                                             pandas.DataFrame()'

        X_new = pd.DataFrame()
        for col in X.columns.values:
            if self.is_numeric(X[col]):
                X_new[col] = self.finger(X[col], 
                                         cut_points=self.bins_[col][1:-1])
            else:
                X_new[col] = X[col]
        
        return X_new

    def sapling(self, X, y, **params):
        """Creates finitely many intervals for a continuous vector using 
        DecisionTreeClassifier optimizing splits with respect to Gini impurity
        criterium.

        Parameters
        ----------
        X: array-like, shape (n_samples, )
            Vector passed as an one-dimensional array-like object where 
            n_samples in the number of samples.

        y: array-like, shape (n_samples, )
            Vector of corresponding to X values passed as an one-dimensional 
            array-like object where n_samples is the number of samples.

        Returns
        -------
        bins: array, shape (n_bins, )
            Vector of successive cut-off points being upper limits of the 
            corresponding intervals as well as containing a minimum value. 

        """
        y = np.asarray(y)
        X = np.asarray(X)
        y = y[~np.isnan(X)]
        X = X[~np.isnan(X)]

        if len(np.unique(X)) < self.min_bins:
            raise RuntimeError('not enough unique values in the array')

        clf = DecisionTreeClassifier(max_depth=self.max_tree_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     max_leaf_nodes=self.max_bins)
        X = X.reshape(-1, 1)
        
        clf.fit(X, y)
        
        min_val = min(X)
        max_val = max(X)
        
        bins = clf.tree_.threshold[clf.tree_.feature != -2]
        np.append(bins, [min_val, max_val])
        bins = np.unique(bins)

        cols_out_idx = []
        for idx, val in enumerate(bins):
            if ~(min_val <= val <= max_val):
                cols_out_idx.append(idx) 

        bins = np.delete(bins, cols_out_idx)

        return bins

    def spearman(self, X, y, **params):
        """Creates finitely many intervals for a continuous vector with use of
        Spearman's rang correlation (supervised).

        Parameters
        ----------
        X: array-like, shape (n_samples, )
            Vector passed as an one-dimensional array-like object where 
            n_samples is the number of samples.

        y: array-like, shape (n_samples, )
            Vector of corresponding to X values passed as an one-dimensional 
            array-like object where n_samples is the number of samples.

        Returns
        -------
        bins: array, shape (n_bins, )
            Vector of successive cut-off points being upper limits of the 
            corresponding intervals as well as containing a minimum value. 

        """
        y = np.asarray(y)
        X = np.asarray(X)
        y = y[~np.isnan(X)]
        X = X[~np.isnan(X)]

        if len(np.unique(X)) < self.min_bins:
            raise RuntimeError('not enough unique values in the array')
        
        r = 0
        n = self.max_bins + 1
        while np.abs(r) < 1:
            bins = algos.quantile(np.unique(X), np.linspace(0, 1, n))
            df = pd.DataFrame({'X': X, 
                               'y': y, 
                               'Bucket': pd.cut(X, bins=bins, 
                                                include_lowest=True)
                              })
            df_gr = df.groupby(by='Bucket', as_index=True)
            if not (df_gr.agg('count').X == 0).any():
                r, p = stats.spearmanr(getattr(df_gr, self.formula)().X,
                                       getattr(df_gr, self.formula)().y) 
                
            if n == self.min_bins + 1:
                break

            n -= 1

        return bins
            
    @staticmethod
    def finger(X, y=None, cut_points=None, 
               n_quantiles=4, labels=None, 
               min_val=None, max_val=None, **params):
        """Manually bins continuous variable into the declared intervals.

        If the cut-off points are not declared the split is made using 
        quantiles.

        Parameters
        ----------
        X: array-like, shape (n_samples, )
            Vector passed as an one-dimensional array-like object where 
            n_samples in the number of samples.

        y: Ignore

        cut_points: array-like
            Increasing monotonic sequence generating right-closed intervals. 
            Values not allocated to any of the categories will be assigned to 
            the empty set. For example given: cut_points=[1, 5, 9] will
            generate intervals: [X.min(), 1], (1, 5], (5, 9], (9, X.max()]. 
            If you want to specify lower and upper limitations, set parameters:
            "min_val", "max_val" to a specific value. 

        n_quantiles: int (default: 4)
            When cut_points are not declared it sets the number of quantiles
            to which the variable will be splitted. For example setting
            n_quantiles = 4 will return quartiles of X values between min_val
            and max_val.

        labels: string {'auto'} or array-like
            Specifies returned bucket names, needs to be the same length as the
            number of created buckets.
            auto:
                Assigns default values to group names by numbering them.
            
        min_val: float
            Determines lower limit value. If not specified takes -np.inf.

        max_val: float
            Determines upper limit value. If not specified takes np.inf.

        Returns
        -------
        X_new: array, shape (n_samples, )
            Input data with its original values ​​being substituted with their 
            respective labels.

        """
        X = np.asarray(X)
        x = X[~np.isnan(X)]
        
        if len(np.unique(x)) <= 1:
            raise RuntimeError('Not enough unique values.')

        if min_val is None:
            min_val = -np.inf

        if max_val is None:    
            max_val = np.inf

        # Default break_points in case of no declaration of cut_points
        if cut_points is None:
            x = x[(x>=min_val) & (x<=max_val)]
            break_points = algos.quantile(np.unique(x), 
                                          np.linspace(0, 1, n_quantiles+1))
        else:    
            break_points = np.insert(cut_points, 
                                     [0, len(cut_points)], 
                                     [min_val, max_val])

        if labels == 'auto':
            labels = range(len(break_points)-1)

        X_new = pd.cut(X, bins=break_points, 
                       labels=labels, include_lowest=True)

        return X_new