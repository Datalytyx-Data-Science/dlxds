"""
Module for preprocessing tools.

Class List (in order):
DataFrameSelector
"""

#Column Selectors
#=================


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Description:
    A selection tranformer that will select columns of specific datatypes,
    e.g. numeric, categoric etc..
    
    
    Authors:
    Chris Schon
    Eden Trainor
    
    TODO:
    Allow for choice to return numpy array or pandas Dataframe
    """
    
    def __init__(self, attribute_names, returning = 'DataFrame'):
        """
        Description
        -----------
        Initialise the transformer object.
        
        Args
        ----
        attribute_names: string
        
        """
        
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        """
        Description
        -----------
        No fitting required for this transformer.
        Simply checks that input is a pandas dataframe.
        
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
        
        assert isinstance(X, pd.DataFrame)
        
        return self
    
    def transform(self, X):
        """
        Description
        -----------
        
        
        Args
        ----
        X: DataFrame, (examples, features)
            Pandas DataFrame containing training or  example features
            
        Returns
        -------
        self: sklearn.transfromer object
            Returns the fitted transformer object.
        """
    
        return X[self.attribute_names].values
    

    
#=======    
#Scalers
#=======

