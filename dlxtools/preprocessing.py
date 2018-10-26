"""
Module for preprocessing tools.
"""

__all__ = [
'DataFrameSelector',
'FromModelFeatureSelector',
'KBestFeatureSelector',
'PCAFeatureSelector',
'RemoveCorrelatedFeatures',
'SparseFeatureDropper',
'NoneReplacer',
'AnyNaNRowRemover',
'DuplicateColumnRemover'
]

#===========================================================================================
#Imports
#===========================================================================================

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler, 


#===========================================================================================
#Column Selectors
#===========================================================================================

class ConstantFeatureDropper(TransformerMixin, BaseEstimator):
    """
    Transformer drops features from DataFrame that 
    """
    
    def __init__(self):
        pass
    
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

class RobustScalerNumerical(RobustScaler):
    """Implements RobustScaler on numerical columns only."""
    
    def fit(self, X, y = None):
        
        """Isolates numerical columns and fits using sklearn.preprocessing.RobustScaler().fit()
        """
        
        #Isolate numerical columns (in secom this is all)
        self.numerical_columns = X.select_dtypes([np.number]).columns
        
        #Call parent fit method on just numerical columns
        super(RobustScalerNumerical, self).fit(X[self.numerical_columns], y)
        
        return self
    
    def transform(self, X):
        """Applies sklearn.preprocessing.RobustScaler().transform() to numerical columns.
        Concatenates scaled numerical and non numerical columns together.
        """
        
        #Scale numerical columns
        X_num = X[self.numerical_columns]
        X_scaled = super(RobustScalerNumerical, self).transform(X_num)
        
        #Move to a dataframe for concatenation, Important to have the same row indicies and column headers. 
        X_scaled_df = pd.DataFrame(X_scaled, columns = self.numerical_columns, index = X.index)
        
        #Gather non numerical columns
        X_not_scaled = X[[column for column in X.columns if column not in self.numerical_columns]]
        
        return pd.concat([X_scaled_df, X_not_scaled], axis = 'columns')[X.columns] #Concatenate columns in original order

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Description:
    ------------
    A selection tranformer that will select columns of specified Pandas datatypes:
    
    Pandas          |Python  |Used For
    ----------------+--------+---------------------------------
    object          |str     |Text
    int64           |int     |Integer numbers
    float64         |float   |Floating point numbers
    bool            |bool    |True/False values
    datetime64      |NA      |Date and time values
    timedelta[ns]   |NA      |Differences between two datetimes
    category        |NA      |Finite list of text values
    
    
    Authors:
    --------
    Chris Schon
    Eden Trainor
    

    TODO:
    -----
    """
    
    def __init__(self, attribute_name):
        """
        Description
        -----------
        Initialise the transformer object.
        
        Args
        ----
        attribute_names: string
            Columns coresponding to this attribute will be selected.
        """
        self.attribute_name = attribute_name
        
 
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
        

        assert isinstance(X, (pd.DataFrame, np.ndarray))
        
        return self
    
    def transform(self, X):
        """
        Description
        -----------
        Selects and returns columns of pandas data type attribute_name
        
        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training or  example features
            
        Returns
        -------
        X_typed: DataFrame, (examples, features)
            Pandas DataFrame containing columns of only  
        """
        
        X_typed = X.select_dtypes(attribute_name)
        
        return X_typed

class SparseFeatureDropper(TransformerMixin, BaseEstimator):
    """
    Transformer drops features with a certain percentage of empty rows. The user can set the threshold for this.
    """
    
    def __init__(self, set_threshold = 90):
        self.set_threshold = set_threshold
        
        
    def fit(self, X, y = None): #has to take an optional y for pipelines
        
        """Calculates the number of missing values the corresponds to the threshold.
        Detects and labels columns with more missing values that the threshold. 
        """
        #Threshold defined by # full bins in df.dropna(), we've defined  threshold as percentage empty bins.
        absolute_threshold = (100 - self.set_threshold)*X.shape[0]/100 
        
        self.drop_columns = features.isna().sum()[features.isna().sum() > absolute_threshold].index #Calculates pd.series with column lables as indecies
        
        return self
    
    def transform(self, X):
        
        """Drops columns with more missing values than the threshold.
        """   
        
        assert isinstance(X, pd.DataFrame)
        
        return X.drop(columns = self.drop_columns) 


#Select from an SKLearn built in model
class FromModelFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Authors
    -------
    Chris Schon
    """

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    def fit(self, X, y):
        self.fromModel = SelectFromModel(estimator=self.model, threshold = self.threshold)
        self.fromModel.fit(X, y)
        return self
    def transform(self, X):
        return self.fromModel.transform(X)


#Use SKLearn SelectKBest for feature selection
class KBestFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Authors
    -------
    Chris Schon
    """
    def __init__(self, k, scorefunc):
        self.k = k
        self.scorefunc = scorefunc
    def fit(self, X, y):
        self.kbest = SelectKBest(score_func = self.scorefunc, k = self.k)
        self.kbest.fit(X, y)
        return self
    def transform(self, X):
        return self.kbest.transform(X)
   



#Select top k Principal Components for feature
class PCAFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Authors
    -------
    Chris Schon
    """
    def __init__(self, k):
        self.k = k
    def fit(self, X, y=None):
        self.pca = TruncatedSVD(n_components = self.k)
        self.pca.fit(X)
        return self
    def transform(self, X):
        return self.pca.transform(X)

    
    
    
class CorrelatedFeatureRemover(BaseEstimator, TransformerMixin):

    """
    A class that drops features if the absolute pairwise correlation between features
    is greater than the specified corr_threshold, therefore both strong negative and
    positive correlations are accounted for. If no corr_threshold is specified then
    the corr_threshold is 0.9.
    
    Parameters
    ----------
    method : {'pearson', 'kendall', 'spearman'}
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
    
    corr_threshold: The threshold above which correlated features should be dropped.
                    Must be between -1 and 1
    
    print_drop_feat:    If "True" then the correlated feature set and the corr val
                        of the dropped features is printed to the screen.
        
    Authors
    -------
    William Holtam
    """
    
    def __init__(self, method = 'pearson', corr_threshold = 0.9, print_drop_feat = False):
        
        """
        Description
        -----------
        Initialise the transformer object and sets the method, 
        corr_threshold and print_drop_featas instance variables.
        """
        
        self.method = method
        self.corr_threshold = corr_threshold
        self.print_drop_feat = print_drop_feat
    
    def fit(self, X, y = None):
        
        """
        Fit creates a correlation matrix and itterates through it to identify
        columns which are correlated to a greater extent than the corr_threshold.
        
        The column numbers of these columns are appended to the "drop_cols" list.
        The "drop_cols" list is sorted and assigned to the instance variable self.drops.
        """
        
        # Creates Correlation Matrix    
        corr_matrix = X.corr()
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
                    if self.print_drop_feat == True:
                        print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    
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


#===========================================================================================
#Data Cleaners
#===========================================================================================

class SparseFeatureDropper(TransformerMixin, BaseEstimator):

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
     
    def __init__(self, set_threshold = 90):
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
        
        
    def fit(self, X, y = None): #has to take an optional y for pipelines
        """
        Description
        -----------
        Calculates the number of missing values the corresponds to the threshold.
        Detects and labels columns with more missing values that the threshold.
        
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


        #Threshold defined by # full bins in df.dropna(), we've defined  threshold as percentage empty bins.
        absolute_threshold = (100 - self.set_threshold)*X.shape[0]/100 
        
        
        self.drop_columns = features.isna().sum()[features.isna().sum() > absolute_threshold].index #Calculates pd.series with column lables as indecies
        
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
            Pandas DataFrame containing example features without columns in drop_columns            
        """   
        
        assert isinstance(X, pd.DataFrame)
        
        X_full = X.drop(columns = self.drop_columns)

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
    
    def __init__(self):
        """
        Description
        -----------
        Initialise the transformer object.
        """
        pass

    def fit(self, X, y = None):
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
        Transform confirms X is a Dataframe and fills Nonetype with pd.np.nan.

        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training/test example features.

        Returns
        -------
        X_full: DataFrame, (examples, features)
            Pandas DataFrame containing example features with np.nan values instead of NoneType objects.            
        """
        
        assert isinstance(X, pd.DataFrame)
        
        X.fillna(value = pd.np.nan, inplace=True)
        
        return pd.DataFrame(X, index = X.index, columns = X.columns)
    
    
    
    
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
        
    def fit(self, X, y = None):  # has to take an optional y for pipelines
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
            Pandas DataFrame containing example features with np.nan values instead of NoneType objects.            
        """
        
        assert isinstance(X, pd.DataFrame)
        
        X = X.dropna(axis=0, how='any')
        
        self.cleaned_data = X
        
        return pd.DataFrame(X, index = X.index, columns = X.columns)

    
    
    
class DuplicateColumnRemover(BaseEstimator, TransformerMixin):
    
    """
    """
    
    def fit(self, X, y = None):
        groups = X.columns.to_series().groupby(X.dtypes).groups
        dups = []
    
        for int64, float64 in groups.items():
    
            columns = X[float64].columns
            vs = X[float64]
            columns_length = len(columns)
            
    
            for i in range(columns_length):
                ia = vs.iloc[:,i].values
                for j in range(i+1, columns_length):
                    ja = vs.iloc[:,j].values
                    if array_equal(ia, ja):
                        dups.append(columns[i])
                        break
                        
        self.dups = dups
        return self
        
    def transform(self, X):
        X = X.drop(self.dups, axis=1)
        return X

    
#===========================================================================================    
#Scalers
#===========================================================================================

class RobustScalerNumerical(RobustScaler):
    """Implements RobustScaler on numerical columns only."""
    
    def fit(self, X, y = None):
        
        """Isolates numerical columns and fits using sklearn.preprocessing.RobustScaler().fit()
        """
        
        #Isolate numerical columns (in secom this is all)
        self.numerical_columns = X.select_dtypes([np.number]).columns
        
        #Call parent fit method on just numerical columns
        super(RobustScalerNumerical, self).fit(X[self.numerical_columns], y)
        
        return self
    
    def transform(self, X):
        """Applies sklearn.preprocessing.RobustScaler().transform() to numerical columns.
        Concatenates scaled numerical and non numerical columns together.
        """
        
        #Scale numerical columns
        X_num = X[self.numerical_columns]
        X_scaled = super(RobustScalerNumerical, self).transform(X_num)
        
        #Move to a dataframe for concatenation, Important to have the same row indicies and column headers. 
        X_scaled_df = pd.DataFrame(X_scaled, columns = self.numerical_columns, index = X.index)
        
        #Gather non numerical columns
        X_not_scaled = X[[column for column in X.columns if column not in self.numerical_columns]]
        
        return pd.concat([X_scaled_df, X_not_scaled], axis = 'columns')[X.columns] #Concatenate columns in original order
    
    
    #------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------
    
    
    
 class CorrelationSelector(TransformerMixin, BaseEstimator):
    
    
    # A Transformer which takes a dataframe and retains those columns which are calucated to be most correlated with target variable.
    
      " Authors " 
        
      # Thomas Rowe 
    
    
     def __init__(self, n_columns = 50):
        
        self.n_columns = n_columns
   

     def fit(self, X, y = None):
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')
        
        #Find correlations
        correlations_target = []

        for i in X.columns:
            
            corr, p_test = sp.stats.spearmanr(X[i], y)
            correlations_target.append(corr)
            
        corr_train_target = pd.Series(correlations_target, index = X.columns).abs()
        
        #Check for nans
        if corr_train_target.isna().sum() != 0: 
            raise ValueError('Feature corelations contain nans from {}'.format(self.__class__))
            
        self.largest = corr_train_target.nlargest(self.n_columns)
        
        #Print a metric
        self.correlation_total = corr_train_target.sum()
        self.correlation_score = self.largest.sum()
        
        self.correlation_percentage = self.correlation_score/self.correlation_total*100
        
        print('The {} features with largest correlation to the labels contain {:.2f}% of the total correlation score.'
              .format(self.n_columns, self.correlation_percentage))
        
        
        self.most_correlated_cols = self.largest.index
            
        return self
                 
    
     def transform(self, X):
        
        
        return X.loc[:, self.most_correlated_cols] 
    
    
    
