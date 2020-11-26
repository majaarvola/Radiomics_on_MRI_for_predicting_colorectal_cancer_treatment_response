import pandas as pd
import pymrmr
from sklearn.preprocessing import KBinsDiscretizer

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
    else:
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
