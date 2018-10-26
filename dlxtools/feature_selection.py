"""
This is the module for feature selction tools
"""
__all__ = [
'DataFrameSelector',
'CorrelationSelector',
'PCAVarThreshSelector',
'FromModelFeatureSelector',
'KBestFeatureSelector',
'PCAFeatureSelector',
'FromModelFeatureSelector',
'KBestFeatureSelector',
'PCAFeatureSelector'
]
#===========================================================================================
#Imports
#===========================================================================================

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#===========================================================================================
#Selectors
#===========================================================================================   

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





class CorrelationSelector(TransformerMixin, BaseEstimator):
    
    """
    Description
    -----------
    A Transformer which takes a dataframe and retains those columns which are calucated to be most correlated with target variable.
    
    Authors
    -------
    Thomas Rowe 
    """
    
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
    
    

class PCAVarThreshSelector(PCA):
    """
    Description
    -----------
    Selects the columns that can explain a certain percentage of the variance in a data set
    
    Authors
    -------
    Eden Trainor
    
    Notes
    -----
    1. PCA has a principole component limit of 4459 components, no matter how many more features you put into
    it this is a hrad limit of how many components it will return to you.
  
    """
    
    def __init__(self, 
                 n_components=None, 
                 copy=True, 
                 whiten=False, 
                 svd_solver='auto', 
                 tol=0.0, 
                 iterated_power='auto', 
                 random_state=None, 
                 explained_variance_thresh = 0.8):
        
        
        super().__init__(n_components, 
                         copy, 
                         whiten, 
                         svd_solver, 
                         tol, 
                         iterated_power, 
                         random_state)

        
        #Set threshold
        self.explained_variance_thresh = explained_variance_thresh
        
        #Check threshold is in valid range
        if not (0 < explained_variance_thresh <= 1):
            raise (ValueError('explained_variance_thresh must be between 0 and 1 (default 0.8), '.format(
                explained_variance_thresh)))                  )
            
        
        
    def find_nearest_index(self, array, value):
        """
        Description
        -----------
        Finds the index of the coefficient in an array nearest a certain value.
        
        
        Args
        ----
        array: np.ndarray, (number_of_componants,)
            Array containing coeffficients 
        
        value: int,
            Index of coefficient in array closset to this value is found.
        
        
        Returns
        -------
        index: int,
            Index of coefficient in array closest to value.
        """
               
        index = (np.abs(array - value)).argmin()
        
        print('{}: {} features are needed to explain {}% of the variance in the data.'.format(
        	self.__class__, 
            index, 
            self.explained_variance_thresh*100))
        
        return index
    
        
    def fit(self, X, y = None):
        """
        Description
        -----------
        Fits the PCA and calculates the index threshold index of the cumulative explained variance ratio array.
        
        
        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training example features
            
        y: array/DataFrame, (examples,)
            (Optional) Training example labels
        
        Returns
        -------
        self: PCAVarThreshSelector instance
            Returns transfromer instance with fitted instance variables on training data.
        """

        assert isinstance(X, pd.DataFrame), 'input isn\'t pandas DataFrame'
        
        #PCA fit the dataset
        super().fit(X)
        
        #Get the cumulative explained variance ratio array (ascending order of cumulative variance explained)
        cumulative_EVR = self.explained_variance_ratio_.cumsum()
        
        #Finds the index corresponding to the threshold amount of variance explained
        self.indx = self.find_nearest_index(array = cumulative_EVR, 
                                            value = self.explained_variance_thresh)
        
        
        return self
    
    def transform(self, X):
        """
        Description
        -----------        
        Selects all the principle components up to the threshold variance.
        
        
        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training example features


        Returns
        -------
        self: np.ndarray, (examples, indx)
            Array containing the minimum number of principle componants required by explained_variance_thresh.
        """
        
        assert isinstance(X, pd.DataFrame)
        
        #Trnasform data into principal componant mode
        all_components =  super().transform(X)
        
        
        
        return pd.DataFrame(all_components[:, :self.indx], index = X.index)
    
    def fit_transform(self, X, y = None):
        """
        Description
        -----------
        Combines fit and transform methods. 
        This is especially required in this class to overwrite the fit_transform in PCA as fit method not called in 
        PCA fit_transform method.
        
        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training example features
            
        y: array/DataFrame, (examples,)
            (Optional) Training example labels            
        
        Returns
        -------
        self: np.ndarray, (examples, indx)
            Array containing the minimum number of principle componants required by explained_variance_thresh.
        """
                            
        return self.fit(X, y).transform(X)


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