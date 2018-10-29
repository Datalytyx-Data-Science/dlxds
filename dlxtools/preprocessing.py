"""
Module for preprocessing tools.
"""

# List in order
__all__ = [
    'SparseFeatureRemover',
    'CorrelatedFeatureRemover',
    'SparseFeatureRemover',
    'NoneReplacer',
    'AnyNaNRowRemover',
    'DuplicateColumnRemover',
    'PandasRobustScaler',
    'DataFrameSelector',
    'ConstantFeatureRemover',
    'PairwiseCorrelationPlotter'
]


# ===========================================================================================
# Imports
# ===========================================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import RobustScaler


# ===========================================================================================
# Data Cleaners
# ===========================================================================================

class SparseFeatureRemover(TransformerMixin, BaseEstimator):
    """
    Description
    -----------
    Transformer drops features with a certain percentage of empty rows.
    The user can set the threshold for this.

    Authors
    -------
    Eden Trainor

    TODO:
    ----
    Error. Returns no columns when threshold is set to 100.
    """

    def __init__(self, set_threshold=100):
        self.set_threshold = set_threshold

    def fit(self, X, y=None):  # has to take an optional y for pipelines

        """
        Calculates the number of missing values the corresponds to the
        threshold. Detects and labels columns with more missing values
        that the threshold.
        """

        # Threshold defined by # full bins in df.dropna(),
        # we've defined threshold as percentage empty bins.
        absolute_threshold = (100 - self.set_threshold)*X.shape[0]/100

        self.drop_columns = (
            features.isna().sum()[
                features.isna().sum()
                > absolute_threshold
            ].index
        )  # Calculates pd.series with column lables as indecies

        return self

    def transform(self, X):

        """
        Drops columns with more missing values than the threshold.
        """

        assert isinstance(X, pd.DataFrame)

        return X.drop(columns=self.drop_columns)


class CorrelatedFeatureRemover(BaseEstimator, TransformerMixin):

    """
    A class that drops features if the absolute pairwise correlation
    between features is greater than the specified corr_threshold,
    therefore both strong negative and positive correlations are accounted
    for. If no corr_threshold is specified then the corr_threshold is 0.9.

    Parameters
    ----------
    method : {'pearson', 'kendall', 'spearman'}
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation

    corr_threshold: The threshold above which correlated features should be
                    dropped. Must be between -1 and 1

    print_drop_feat: If "True" then the correlated feature set and the
                     corr val of the dropped features is printed to the screen.

    Authors
    -------
    William Holtam
    """

    def __init__(
        self,
        method='pearson',
        corr_threshold=0.9,
        print_drop_feat=False
    ):

        """
        Description
        -----------
        Initialise the transformer object and sets the method,
        corr_threshold and print_drop_featas instance variables.
        """

        self.method = method
        self.corr_threshold = corr_threshold
        self.print_drop_feat = print_drop_feat

    def fit(self, X, y=None):

        """
        Fit creates a correlation matrix and itterates through it to
        identify columns which are correlated to a greater extent than
        the corr_threshold.

        The column numbers of these columns are appended to the "drop_cols"
        list. The "drop_cols" list is sorted and assigned to the instance
        variable self.drops.
        """

        # Creates Correlation Matrix
        corr_matrix = X.corr(method=self.method)
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []
        count = 0

        # Iterates through Correlation Matrix Table to find correlated columns
        for i in iters:
            for j in range(i):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = item.values

                if abs(val) >= self.corr_threshold:

                    # Prints the correlated feature set and the corr val
                    if self.print_drop_feat is True:

                        print(
                            col.values[0],
                            "|",
                            row.values[0],
                            "|",
                            round(val[0][0], 2)
                        )

                    drop_cols.append(i)
                    count += 1

        print(str(count)+" features have been droped.")
        self.drops = sorted(set(drop_cols))[::-1]

        return self

    def transform(self, X):

        """
        Transform indexes the inputed dataframe X for the dropped columns and
        drops them from the dataframe.
        """

        # Drops the correlated columns
        for i in self.drops:
            col = X.iloc[:, (i+1):(i+2)].columns.values
            X = X.drop(col, axis=1)

        return X


class SparseFeatureRemover(TransformerMixin, BaseEstimator):

    """
    Description:
    ------------
    Transformer drops feature columns with a certain percentage of empty rows.
    The user can set the threshold for this.


    Authors:
    --------
    Eden Trainor

    TODO:
    -----

    """

    def __init__(self, set_threshold=90):
        """
        Description
        -----------
        Initialise the transformer object.

        Args
        ----
        set_threshold: int
            Intiger percentage of empty empty rows in a column.
        """

        self.set_threshold = set_threshold

    def fit(self, X, y=None):  # has to take an optional y for pipelines
        """
        Description
        -----------
        Calculates the number of missing values the corresponds to the
        threshold. Detects and labels columns with more missing values
        that the threshold.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training example features

        y: array/DataFrame, (examples,)
            (Optional) Training example labels

        Returns
        -------
        self: sklearn.transfromer object
            Returns the fitted transformer object.
        """

        # Threshold defined by # full bins in df.dropna(),
        # we've defined threshold as percentage empty bins.
        absolute_threshold = (100 - self.set_threshold)*X.shape[0]/100

        self.drop_columns = (
            features.isna().sum()[
                features.isna().sum() > absolute_threshold
            ].index
        )  # Calculates pd.series with column lables as indecies

        return self

    def transform(self, X):
        """
        Description
        -----------
        Drops columns with more missing values than the threshold.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training/test example features

        Returns
        -------
        X_full: DataFrame, (examples, features)
            Pandas DataFrame containing example features without columns
            in drop_columns.
        """

        assert isinstance(X, pd.DataFrame)

        X_full = X.drop(columns=self.drop_columns)

        return X_full


class NoneReplacer(TransformerMixin, BaseEstimator):

    """
    Description:
    ------------
    Transformer changes Nonetype values into numpy NaN values.

    Authors:
    --------
    William Holtam

    TODO:
    -----

    """

    def fit(self, X, y=None):
        """
        Description
        -----------
        No fitting required for this transformer.
        Simply checks that input is a pandas dataframe or a numpy array.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training example features

        y: array/DataFrame, (examples,)
            (Optional) Training example labels

        Returns
        -------
        self: sklearn.transfromer object
            Returns the fitted transformer object.
        """
        return self

    def transform(self, X):
        """
        Description
        -----------
        Transform confirms X is a DataFrame and fills Nonetype with pd.np.nan.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training/test example features.

        Returns
        -------
        X_full: DataFrame, (examples, features)
            Pandas DataFrame containing example features with np.nan values
            instead of NoneType objects.
        """

        assert isinstance(X, pd.DataFrame)

        X.fillna(value=pd.np.nan, inplace=True)

        return pd.DataFrame(X, index=X.index, columns=X.columns)


class AnyNaNRowRemover(TransformerMixin, BaseEstimator):
    """
    Description:
    ------------
    Transformer drops any rows where where any element in row is NaN.

    Authors:
    --------
    William Holtam

    TODO:
    -----

    """

    def __init__(self):
        """
        Description
        -----------
        Initialise the transformer object.
        """
        pass

    def fit(self, X, y=None):  # has to take an optional y for pipelines
        """
        Description
        -----------
        No fitting required for this transformer.
        Simply checks that input is a pandas dataframe or a numpy array.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training example features

        y: array/DataFrame, (examples,)
            (Optional) Training example labels

        Returns
        -------
        self: sklearn.transfromer object
            Returns the fitted transformer object.
        """
        return self

    def transform(self, X):
        """
        Description
        -----------
        Transform drops any rows where where any element in row is NaN.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training/test example features.

        Returns
        -------
        X_full: DataFrame, (examples, features)
            Pandas DataFrame containing example features with np.nan values
            instead of NoneType objects.
        """

        assert isinstance(X, pd.DataFrame)

        X = X.dropna(axis=0, how='any')

        self.cleaned_data = X

        return pd.DataFrame(X, index=X.index, columns=X.columns)


class DuplicateColumnRemover(BaseEstimator, TransformerMixin):

    """
    """

    def fit(self, X, y=None):
        groups = X.columns.to_series().groupby(X.dtypes).groups
        dups = []

        for int64, float64 in groups.items():

            columns = X[float64].columns
            vs = X[float64]
            columns_length = len(columns)

            for i in range(columns_length):
                ia = vs.iloc[:, i].values
                for j in range(i+1, columns_length):
                    ja = vs.iloc[:, j].values
                    if array_equal(ia, ja):
                        dups.append(columns[i])
                        break

        self.dups = dups
        return self

    def transform(self, X):
        X = X.drop(self.dups, axis=1)
        return X


class ConstantFeatureRemover(TransformerMixin, BaseEstimator):

    """
    Transformer drops features from DataFrame that
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):

        # Isolate numerical columns (in secom this is all)
        numerical_columns = X.select_dtypes([np.number]).columns

        # Calculatet the standard deviation of numerical columns
        standard_deviation = X[numerical_columns].std()

        # Indicate which columns have no standard deviation
        self.columns_to_drop = standard_deviation[
            standard_deviation == 0
        ].index

        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis='columns')


# ===========================================================================================
# Scalers
# ===========================================================================================


class PandasRobustScaler(RobustScaler):
    """
    Description
    -----------
    Implements a robust scale and returns a pandas DataFrame.

    Authors
    -------
    Eden Trainor

    Notes
    -----
    1. In most cases this class will need to be used with a selector
    class in a FeatureUnion.
    """

    def fit(self, X, y=None):
        """
        Description
        -----------
        Simple wrapper around RobustScaler Fit to check the fit is on
        Pandas DataFrame
        """
        assert (
            isinstance(X, pd.DataFrame),
            '{}: input into fit method must be a pandas DataFrame'.format(
                self.__class__
            )
        )

        super().fit(X, y)

        return self

    def transform(self, X):
        """
        Description
        -----------
        Simple wrapper around the RobustScaler Transform method.
        """
        assert (
            isinstance(X, pd.DataFrame),
            '{}: input into transform method must be a pandas DataFrame'.format(
                self.__class__
            )
        )

        df_output = pd.DataFrame(
            super().transform(X),
            index=X.index,
            columns=X.columns
        )

        return df_output


class PandasNormalizer(Normalizer):
	"""
	Description
	-----------
	Implements a Normalizer scale and returns a pandas DataFrame.

	Authors
	-------
	Thomas Rowe

	Notes
	-----
	1. In most cases this class will need to be used with a selector class in a FeatureUnion.
	"""

	def fit(self, X, y=None):
		"""
		Description
		-----------
		Simple wrapper around  Fit to check the fit is on Pandas DataFrame
		"""
		assert (
			isinstance(X, pd.DataFrame),
			'{}: input into fit method must be a pandas DataFrame'.format(self.__class__)
			)

		super().fit(X, y)

		return self

	def transform(self, X):
		"""
		Description
		-----------
		Simple wrapper around the  Transform method.
		"""
		assert (
			isinstance(X, pd.DataFrame),
			'{}: input into transform method must be a pandas DataFrame'.format(self.__class__)
			)

		return pd.DataFrame(super().transform(X), index=X.index, columns=X.columns)


class PandasStandardScaler(StandardScaler):
	"""
	Description
	-----------
	Implements a standard scale and returns a pandas DataFrame.

	Authors
	-------
	Thomas Rowe

	Notes
	-----
	1. In most cases this class will need to be used with a selector class in a FeatureUnion.
	"""

	def fit(self, X, y=None):
		"""
		Description
		-----------
		Simple wrapper around  Fit to check the fit is on Pandas DataFrame
		"""
		assert (
			isinstance(X, pd.DataFrame),
			'{}: input into fit method must be a pandas DataFrame'.format(self.__class__)
			)

		super().fit(X, y)

		return self

	def transform(self, X):
		"""
		Description
		-----------
		Simple wrapper around the  Transform method.
		"""
		assert (
			isinstance(X, pd.DataFrame),
			'{}: input into transform method must be a pandas DataFrame'.format(self.__class__)
			)

		return pd.DataFrame(super().transform(X), index=X.index, columns=X.columns)


class PandasMinMaxScaler(Normalizer):
	"""
	Description
	-----------
	Implements a MinMaxScaler scale and returns a pandas DataFrame.

	Authors
	-------
	Thomas Rowe

	Notes
	-----
	1. In most cases this class will need to be used with a selector class in a FeatureUnion.
	"""

	def fit(self, X, y=None):
		"""
		Description
		-----------
		Simple wrapper around  Fit to check the fit is on Pandas DataFrame
		"""
		assert (
			isinstance(X, pd.DataFrame),
			'{}: input into fit method must be a pandas DataFrame'.format(self.__class__)
			)

		super().fit(X, y)

		return self

	def transform(self, X):
		"""
		Description
		-----------
		Simple wrapper around the  Transform method.
		"""
		assert (
			isinstance(X, pd.DataFrame),
			'{}: input into transform method must be a pandas DataFrame'.format(self.__class__)
			)

		return pd.DataFrame(super().transform(X), index=X.index, columns=X.columns)


# ===========================================================================================
# Visualisations
# ===========================================================================================


class PairwiseCorrelationPlotter(TransformerMixin, BaseEstimator):

    """
    Description:
    ------------
    A class that produces a pairwise correlation plot for features.
    This plot is placed into a Transformer class so that it can be
    included in a sklearn Pipeline.

    Authors:
    --------
    William Holtam

    TODO:
    -----
    """

    def __init__(
        self,
        figsize=(11, 9),
        h_neg=240,
        h_pos=10,
        as_cmap=True
    ):

        """
        Description
        -----------
        Initialise the transformer object and sets the following
        initiation variables:
        * figsize
        * h_neg
        * h_pos
        * as_cmap
        """

        self.figsize = figsize
        self.h_neg = h_neg
        self.h_pos = h_pos
        self.as_cmap = as_cmap

    def fit(self, X, y=None):

        """
        A correlation matrix is created for the X dataframe.
        A mask is created where the correlation matrix is 0,
        the result is True, everywhere else, the result is False.
        Data in the plot will not be shown in cells whre mask is True.

        The result is a heatmap which displays the results of the
        correlation matrix visually, where dark red is a strong positive
        correlation and dark blue is a strongly negative correlation.
        """

        self.correlation = X.corr()
#         print(self.corr)

        # Generate a mask for the upper triangle
        self.mask = np.zeros_like(self.correlation, dtype=np.bool)
        self.mask[np.triu_indices_from(self.mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        self.cmap = sns.diverging_palette(
            h_neg=self.h_neg,
            h_pos=self.h_pos,
            as_cmap=self.as_cmap
        )

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            self.correlation,
            mask=self.mask,
            cmap=self.cmap,
            vmax=.3,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5}
        )

        plt.show()

        return self

    def transform(self, X):

        """
        Returns the passed pandas DataFrame.
        """

        return X