import pandas as pd
import pymrmr
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def select_features(method, params, selectionFeaturesPath, manualFeaturesPath): 
    """
    ACTION: 
        Select features to use for machine learning. Method is specified as an input.
    INPUTS: 
        method: feature selection algorithm to use, eg. 'MRMR'
        nFeatures: number of features to select
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


    if method == 'MRMR':
        # discretization(X)
        XandY = pd.merge(y, X, how='inner', on='id')
        nFeatures = params['nFeatures']
        internalFEMethod = params['internalFEMethod']
        return pymrmr.mRMR(XandY, internalFEMethod, nFeatures)
    else:
        print(f'Method "{method}" is not implemented in feature_selection.py')

    return []


def simple_discretization(X, nBins):
    normalizedData = (nBins-1)*(data-data.min())/(data.max()-data.min())
    return normalizedData.round().astype(int)

# def discretization(X):
#     """
#     docstring
#     """
#     kmeans = KMeans(n_clusters=3, random_state=0).fit(X.values[:,1].reshape(-1,1))
#     print(X.values[:,1])
#     print(kmeans.labels_)
#     plt.plot(kmeans.labels_, X.values[:,1],'or')
#     plt.show()

    # _, nTotalFeatures = X.values.shape
    # for i in range(nTotalFeatures):
    #     kmeans = KMeans(n_clusters=3, random_state=0).fit(X.values[:,i].reshape(-1,1))
    #     X.values[:,i] = kmeans.labels_
    # print(X)
