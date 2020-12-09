import numpy as np
import pandas as pd
import pymrmr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

def select_features(method, params, selectionFeaturesPath, manualFeaturesPath): 
    """
    ACTION: 
        Select features to use for machine learning. Method is specified as an input.
    INPUTS: 
        method: feature selection algorithm to use, eg. 'MRMR'
        params: parameter settings for the selected method (a dictionary)
                For MRMR: nFeatures, internalFEMethod, nBins, discStrategy
        selectionFeaturesPath: path to selectionFeatures file
        manualFeaturesPath: path to manualFeatures file
    OUTPUT:
        List of selected features
    """

    # Read data from csv-files
    X = pd.read_csv(selectionFeaturesPath, index_col=0, delimiter=';') # All data in selectionFeatures.csv
    y = pd.read_csv(manualFeaturesPath, index_col=0, delimiter=';') # All data in manualFeatures.csv

    # Remove test data before doing feature selection
    testIds = [1, 8, 13, 20, 40, 44, 49, 55]
    trainIds = [v for v in X.index.values if v not in testIds]
    y = y.loc[trainIds]
    X = X.loc[trainIds]

    # Drop useless information
    X_diagnostics = [col for col in X if col.startswith('diagnostics')]
    X = X.drop(columns=X_diagnostics) # Data in selectionFeatures.csv, excluding diagnostic features
    X.index.names = ['id'] # Renaming 'patientId' to 'id'
    y = y.filter(items=['outcome']) # Keep only outcome

    # Entering the selection method specified
    if method == 'MRMR':
        # Extracting parameter settings for MRMR
        nFeatures = params['nFeatures']
        internalFEMethod = params['internalFEMethod']
        nBins = params['nBins']
        discStrategy = params['discStrategy']

        X = discretization(X, nBins, discStrategy) # Discretizing data
        XandY = pd.merge(y, X, how='inner', on='id') # Merging input and output into one DataFrame
        return pymrmr.mRMR(XandY, internalFEMethod, nFeatures) # Run MRMR

    elif method == 'LASSO': 
        # Extract parameter setting, fit LASSO model and collect the importance of each feature
        nFeatures = params['nFeatures']
        clf = LassoCV(normalize=True, max_iter=2000).fit(X, y.values.ravel())
        importance = np.abs(clf.coef_)

        # Check that the features selected will have non-zero weight
        nrNonZero = np.count_nonzero(importance)
        if nrNonZero < nFeatures:
            print(f'Warning: Number of features selected with LASSO is reduced from {nFeatures} to {nrNonZero} to only select features with non-zero weights. ')
            nFeatures = nrNonZero
        
        # Select and return the best features (as string names)
        idxFeatures = (-importance).argsort()[:nFeatures]
        return [v for v in X.columns.values[idxFeatures]]
        
    print(f'Method "{method}" is not implemented in feature_selection.py')
    return []


def discretization(X, nBins, discStrategy):
    """
    ACTION: 
        Discretizes the data
    INPUTS: 
        X: DataFrame with input data
        nBins: Number of bins of the discretization
        discStrategy: Strategy for specifying the bins (e.g 'uniform', 'quantile', 'kmeans')
    OUTPUT: 
        New DataFrame with discretized values. 
    """
    trans = KBinsDiscretizer(n_bins=nBins, encode='ordinal', strategy=discStrategy)
    XnewValues = trans.fit_transform(X.values)
    return pd.DataFrame(data=XnewValues, index=X.index, columns=X.columns)
