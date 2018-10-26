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
    'ConstantFeatureRemover'
]

# ===========================================================================================
# Imports
# ===========================================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, SelectFromModel
<<<<<<< HEAD
from sklearn.decomposition import TruncatedSVD
=======
from sklearn.decomposition import TruncatedSVD, PCA
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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

<<<<<<< HEAD
    def __init__(self, set_threshold = 100):
        self.set_threshold = set_threshold


    def fit(self, X, y = None): #has to take an optional y for pipelines

        """Calculates the number of missing values the corresponds to the threshold.
        Detects and labels columns with more missing values that the threshold.
        """
        #Threshold defined by # full bins in df.dropna(), we've defined  threshold as percentage empty bins.
        absolute_threshold = (100 - self.set_threshold)*X.shape[0]/100

        self.drop_columns = features.isna().sum()[features.isna().sum() > absolute_threshold].index #Calculates pd.series with column lables as indecies
=======
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
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

        return self

    def transform(self, X):

<<<<<<< HEAD
        """Drops columns with more missing values than the threshold.
=======
        """
        Drops columns with more missing values than the threshold.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
        """

        assert isinstance(X, pd.DataFrame)

<<<<<<< HEAD
        return X.drop(columns = self.drop_columns)

=======
        return X.drop(columns=self.drop_columns)
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07


class CorrelatedFeatureRemover(BaseEstimator, TransformerMixin):

    """
<<<<<<< HEAD
    A class that drops features if the absolute pairwise correlation between features
    is greater than the specified corr_threshold, therefore both strong negative and
    positive correlations are accounted for. If no corr_threshold is specified then
    the corr_threshold is 0.9.
=======
    A class that drops features if the absolute pairwise correlation
    between features is greater than the specified corr_threshold,
    therefore both strong negative and positive correlations are accounted
    for. If no corr_threshold is specified then the corr_threshold is 0.9.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

    Parameters
    ----------
    method : {'pearson', 'kendall', 'spearman'}
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation

<<<<<<< HEAD
    corr_threshold: The threshold above which correlated features should be dropped.
                    Must be between -1 and 1

    print_drop_feat:    If "True" then the correlated feature set and the corr val
                        of the dropped features is printed to the screen.
=======
    corr_threshold: The threshold above which correlated features should be
                    dropped. Must be between -1 and 1

    print_drop_feat: If "True" then the correlated feature set and the
                     corr val of the dropped features is printed to the screen.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

    Authors
    -------
    William Holtam
    """

<<<<<<< HEAD
    def __init__(self, method = 'pearson', corr_threshold = 0.9, print_drop_feat = False):
=======
    def __init__(
        self,
        method='pearson',
        corr_threshold=0.9,
        print_drop_feat=False
    ):
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

        """
        Description
        -----------
        Initialise the transformer object and sets the method,
        corr_threshold and print_drop_featas instance variables.
        """

        self.method = method
        self.corr_threshold = corr_threshold
        self.print_drop_feat = print_drop_feat

<<<<<<< HEAD
    def fit(self, X, y = None):

        """
        Fit creates a correlation matrix and itterates through it to identify
        columns which are correlated to a greater extent than the corr_threshold.

        The column numbers of these columns are appended to the "drop_cols" list.
        The "drop_cols" list is sorted and assigned to the instance variable self.drops.
        """

        # Creates Correlation Matrix
        corr_matrix = X.corr(method = self.method)
=======
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
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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
<<<<<<< HEAD
                    if self.print_drop_feat == True:
                        print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
=======
                    if self.print_drop_feat is True:

                        print(
                            col.values[0],
                            "|",
                            row.values[0],
                            "|",
                            round(val[0][0], 2)
                        )
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

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
<<<<<<< HEAD

        return X
=======
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

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

<<<<<<< HEAD
    def __init__(self, set_threshold = 90):
=======
    def __init__(self, set_threshold=90):
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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

<<<<<<< HEAD

    def fit(self, X, y = None): #has to take an optional y for pipelines
        """
        Description
        -----------
        Calculates the number of missing values the corresponds to the threshold.
        Detects and labels columns with more missing values that the threshold.
=======
    def fit(self, X, y=None):  # has to take an optional y for pipelines
        """
        Description
        -----------
        Calculates the number of missing values the corresponds to the
        threshold. Detects and labels columns with more missing values
        that the threshold.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

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

<<<<<<< HEAD
        #Threshold defined by # full bins in df.dropna(), we've defined  threshold as percentage empty bins.
        absolute_threshold = (100 - self.set_threshold)*X.shape[0]/100


        self.drop_columns = features.isna().sum()[features.isna().sum() > absolute_threshold].index #Calculates pd.series with column lables as indecies

        return self


=======
        return self

>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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
<<<<<<< HEAD
            Pandas DataFrame containing example features without columns in drop_columns
=======
            Pandas DataFrame containing example features without columns
            in drop_columns.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
        """

        assert isinstance(X, pd.DataFrame)

<<<<<<< HEAD
        X_full = X.drop(columns = self.drop_columns)
=======
        X_full = X.drop(columns=self.drop_columns)
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

        return X_full


<<<<<<< HEAD

=======
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
class NoneReplacer(TransformerMixin, BaseEstimator):

    """
    Description:
    ------------
    Transformer changes Nonetype values into numpy NaN values.

<<<<<<< HEAD

=======
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
    Authors:
    --------
    William Holtam

    TODO:
    -----

    """

<<<<<<< HEAD
    def fit(self, X, y = None):
=======
    def fit(self, X, y=None):
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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
<<<<<<< HEAD
            Pandas DataFrame containing example features with np.nan values instead of NoneType objects.
=======
            Pandas DataFrame containing example features with np.nan values
            instead of NoneType objects.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
        """

        assert isinstance(X, pd.DataFrame)

<<<<<<< HEAD
        X.fillna(value = pd.np.nan, inplace=True)

        return pd.DataFrame(X, index = X.index, columns = X.columns)


=======
        X.fillna(value=pd.np.nan, inplace=True)

        return pd.DataFrame(X, index=X.index, columns=X.columns)
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07


class AnyNaNRowRemover(TransformerMixin, BaseEstimator):
    """
    Description:
    ------------
    Transformer drops any rows where where any element in row is NaN.

<<<<<<< HEAD

=======
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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

<<<<<<< HEAD
    def fit(self, X, y = None):  # has to take an optional y for pipelines
=======
    def fit(self, X, y=None):  # has to take an optional y for pipelines
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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
<<<<<<< HEAD
            Pandas DataFrame containing example features with np.nan values instead of NoneType objects.
=======
            Pandas DataFrame containing example features with np.nan values
            instead of NoneType objects.
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
        """

        assert isinstance(X, pd.DataFrame)

        X = X.dropna(axis=0, how='any')

        self.cleaned_data = X
<<<<<<< HEAD

        return pd.DataFrame(X, index = X.index, columns = X.columns)


=======

        return pd.DataFrame(X, index=X.index, columns=X.columns)
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07


class DuplicateColumnRemover(BaseEstimator, TransformerMixin):

    """
    """

<<<<<<< HEAD
    def fit(self, X, y = None):
=======
    def fit(self, X, y=None):
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
        groups = X.columns.to_series().groupby(X.dtypes).groups
        dups = []

        for int64, float64 in groups.items():

            columns = X[float64].columns
            vs = X[float64]
            columns_length = len(columns)

<<<<<<< HEAD

=======
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
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
<<<<<<< HEAD

    def fit(self, X, y = None):

        #Isolate numerical columns (in secom this is all)
        numerical_columns = X.select_dtypes([np.number]).columns

        #calculatet the standard deviation of numerical columns
        standard_deviation = X[numerical_columns].std()

        #Indicate which columns have no standard deviation
        self.columns_to_drop = standard_deviation[standard_deviation == 0].index

        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis = 'columns')


#===========================================================================================
#Scalers
#===========================================================================================
=======

    def fit(self, X, y=None):
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07

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

<<<<<<< HEAD
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

	def fit(self, X, y = None):
		"""
		Description
		-----------
		Simple wrapper around  Fit to check the fit is on Pandas DataFrame
		"""
		assert isinstance(X, pd.DataFrame), '{}: input into fit method must be a pandas DataFrame'.format(self.__class__)

		super().fit(X, y)

		return self

	def transform(self, X):
		"""
		Description
		-----------
		Simple wrapper around the  Transform method.
		"""
		assert isinstance(X, pd.DataFrame), '{}: input into transform method must be a pandas DataFrame'.format(self.__class__)

		return pd.DataFrame(super().transform(X), index = X.index, columns = X.columns)


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

	def fit(self, X, y = None):
		"""
		Description
		-----------
		Simple wrapper around  Fit to check the fit is on Pandas DataFrame
		"""
		assert isinstance(X, pd.DataFrame), '{}: input into fit method must be a pandas DataFrame'.format(self.__class__)

		super().fit(X, y)

		return self

	def transform(self, X):
		"""
		Description
		-----------
		Simple wrapper around the  Transform method.
		"""
		assert isinstance(X, pd.DataFrame), '{}: input into transform method must be a pandas DataFrame'.format(self.__class__)

		return pd.DataFrame(super().transform(X), index = X.index, columns = X.columns)




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

	def fit(self, X, y = None):
		"""
		Description
		-----------
		Simple wrapper around  Fit to check the fit is on Pandas DataFrame
		"""
		assert isinstance(X, pd.DataFrame), '{}: input into fit method must be a pandas DataFrame'.format(self.__class__)

		super().fit(X, y)

		return self

	def transform(self, X):
		"""
		Description
		-----------
		Simple wrapper around the  Transform method.
		"""
		assert isinstance(X, pd.DataFrame), '{}: input into transform method must be a pandas DataFrame'.format(self.__class__)

		return pd.DataFrame(super().transform(X), index = X.index, columns = X.columns)
=======
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
>>>>>>> aca890d61f00d68e2dd95e58e45577ac3dc3ea07
