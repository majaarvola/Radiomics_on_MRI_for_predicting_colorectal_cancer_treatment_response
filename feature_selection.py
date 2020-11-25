import pandas as pd
import pymrmr


def select_features(method, nFeatures, selectionFeaturesPath, manualFeaturesPath): 
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

# Impement functions for the different feature selection methods
# OUTPUT: feature dictionary (1 if selected, else 0)

featuresPath = "../../patient_data/selection_features_play.csv"
manualFeaturesPath = "../../patient_data/manual_features.csv"

X = pd.read_csv(featuresPath, index_col=0, delimiter=';')
y = pd.read_csv(manualFeaturesPath, index_col=0, delimiter=';')
X_diagnostics = [col for col in X if col.startswith('diagnostics')]
X = X.drop(columns=X_diagnostics)

y['y_patFilt'] = [patId in X.index for patId in y.index.values]
y = y[y['y_patFilt']==True]
y = y.drop(columns=['age', 'y_patFilt'])

print(X)
print(y)

X.index.names = ['id']
# print(X)
merged = pd.merge(y, X, how='inner', on='id')
# print(merged.values)

features = pymrmr.mRMR(merged, "MID", 10)

print(features)