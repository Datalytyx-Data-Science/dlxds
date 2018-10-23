"""
This is a pipelines module

Use for creating and saving commonly used pipelines.
"""
#===========================================================================================
#Imports
#===========================================================================================

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse

#===========================================================================================
#Pipeline construction tools
#===========================================================================================


class PandasFeatureUnion(FeatureUnion):
    
    """
    A DataFrame estimator that applies a list of transformer objects in parallel to the input data,
    then concatenates the results. This is useful to combine several feature extraction mechanisms
    into a single transformer.
    
    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples

    List of transformer objects to be applied to the data. The first half of each tuple
    is the name of the transformer.

    
    n_jobs : int or None, optional (default=None)

    Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
    -1 means using all processors. See Glossary for more details.
    
    
    transformer_weights : dict, optional

    Multiplicative weights for features per transformer. Keys are transformer names,
    values the weights.

        
    Authors
    -------
    Guy who wrote this one:
    Blog: https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union
    Github: https://github.com/marrrcin/pandas-feature-union
    """ 
    
    def fit_transform(self, X, y=None, **fit_params):
        
        """
        Fit all transformers, transform the data and concatenate results.
        
        Parameters
        ----------
        X : Pandas DataFrame only
            Input data to be transformed. 
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
            
        Returns
        -------
        X_t : Pandas DataFrame
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight,
                                        **fit_params)
            for name, trans, weight in self._iter())
        
        
        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        
        
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        
        
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
            
        else:
            Xs = self.merge_dataframes_by_column(Xs)
            
        return Xs
    

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)
    

    def transform(self, X):
        
        """
        Transform X separately by each transformer, concatenate results.
        
        Parameters
        ----------
        X : Pandas DataFrame only
            Input data to be transformed.
            
        Returns
        -------
        X_t : Pandas DataFrame Only
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
            
        else:
            Xs = self.merge_dataframes_by_column(Xs)
            
        return Xs
    

#===========================================================================================    
#Custom Prebuilt Transformers
#===========================================================================================
