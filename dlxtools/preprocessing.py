"""
Module for preprocessing tools.

Class List (in order):
DataFrameSelector
"""
#===========================================================================================
#Column Selectors
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

    
#===========================================================================================    
#Scalers
#===========================================================================================

