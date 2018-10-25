"""
This is the module for exploritory data analyisis tools.
"""
__all__ = ['DFInfo',
			'PCAVarThreshSelector'
			]
#===========================================================================================
#Imports
#===========================================================================================

import pandas as pd



#===========================================================================================
#Descriptive Classes
#===========================================================================================


# Class to extract information from the train and test datasets.
class DFInfo:

    """
    A Class that extracts characteristics of the train and test datasets.
    
    Parameters
    ----------
    train : the train dataset in pandas DataFrame form.
    
    test : the test dataset in pandas DataFrame form.
        
    Authors
    -------
    William Holtam
    """ 
    
    def __init__(self, train, test):
        """
        Description
        -----------
        Initialise the transformer object and sets the train and test datasets as instance variables.
        
        Args
        ----
        train: Pandas DataFrame
            training data
            
        test: Pandas DataFrame
            testing data
        """
    
        self.train = train
        self.test = test
        return
    
    def info(self):
    
        """
        Method extracts the following characteristics from the train and test dataframes:
        * Nº of rows and colums
        * Type of columns
        * Nº of columns with missing values
        * Nº of columns with all rows zero
        """
    
        # Nº of rows and colums
        print('Train: Rows - '+str(len(self.train)) + ' Columns - ' + str(len(self.train.columns)))
        print('Test: Rows - '+str(len(self.test)) + ' Columns - ' + str(len(self.test.columns)))
        
        # Type of columns
        train_col_types = self.train.dtypes
        test_col_types = self.train.dtypes
        print('-'*60)
        print('Train: Type of columns')
        print('-'*60)
        print(train_col_types.groupby(train_col_types).count())
        print('-'*60)
        print('Test: Type of columns')
        print('-'*60)
        print(test_col_types.groupby(test_col_types).count())
        
        # Missing values?
        print('-'*60)
        list = []
        counts = []
        for i in self.train.columns:
            list.append(i)
            counts.append(sum(self.train[i].isnull()))
        print('Train: Nº of columns with missing values')
        print('-'*60)
        print(sum(counts))
        print('-'*60)
        list = []
        counts = []
        for i in self.test.columns:
            list.append(i)
            counts.append(sum(self.test[i].isnull()))
        print('Test: Nº of columns with missing values')
        print('-'*60)
        print(sum(counts))
        
        # Zero Rows
        print('-'*60)
        columns_train_sum = pd.DataFrame(self.train.sum(),columns=['Sum of Row'])
        print('Train: Nº of columns with all rows zero: ')
        print('-'*60)        
        print(str(columns_train_sum[columns_train_sum==0].count()))
        print('-'*60)
        columns_test_sum = pd.DataFrame(self.test.sum(),columns=['Sum of Row'])
        print('Test: Nº of columns with all rows zero: ')
        print('-'*60)        
        print(str(columns_test_sum[columns_train_sum==0].count()))
        print('-'*60)

#===========================================================================================
#Pipeline Classes
#===========================================================================================

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
  

  	TODO
  	----
  	Solve Error with scoring when in pipeline. Detailed explanation:
  	https://stackoverflow.com/questions/52989405/attributeerror-when-scoring-sklearn-pipeline-with-custom-transformer-subclass-bu
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
        
        
        super(PCAVarThreshSelector, self).__init__(n_components, copy, whiten, svd_solver, tol, iterated_power, random_state)

        
        self.explained_variance_thresh = explained_variance_thresh
        
        
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
        
        #PCA fit the dataset
        super(PCAVarThreshSelector, self).fit(X)
        
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
        
        all_components =  super(PCAVarThreshSelector, self).transform(X) #To the sklean limit
        
        return all_components[:, :self.indx]
        