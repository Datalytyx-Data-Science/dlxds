"""
This is the module for exploritory data analyisis tools.
"""

#===========================================================================================
#Imports
#===========================================================================================

import pandas as pd

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
