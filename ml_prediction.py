import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics


def create_evaluate_model(method, params, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams, scoringOptiMetric = 'r2'):
    """
    ACTION: 
        Computes the best parameter setting (if optimizeParams is True) by searching the grid of parameter settings specified by the params-dictionary. 
        Train a model on the train data with given parameters and evaluate it on the test data.  
    INPUTS: 
        method: machine learning algorithm to use, eg. 'RFreg' (random forest regression), 'RFclass' (random forest classifier)
        params: parameter settings for the selected method (a dictionary, values can be lists)
                For RFreg: n_estimators, max_features, max_depth
                For RFclass: n_estimators, max_features, max_depth
                For LogReg: penalty, solver, C, max_iter
        selectedFeatures: list of features to use when traning a model
        selectionFeaturesPath: path to selectionFeatures file
        manualFeaturesPath: path to manualFeatures file
        paramSearchResultsPath: path to paramSearchResults file
        optimizeParams: boolean, if True, GridSearchCV is used to find the best parameter setting
        scoringOptiMetric: metric to optimize over the given set of parameters
    OUTPUTS: 
        yTrueTest: Numpy-array with true outcome values of the test data
        yPredRegTest: Numpy-array with predicted regression outcome values of the test data
        yTrueVal: Numpy-array with true outcome values of the validation data
        yPredRegVal: Numpy-array with predicted regression outcome values of the validation data
        params: The parameter settings that was used on the validation and test data
    """

    # Read input data from csv-files
    X = pd.read_csv(selectionFeaturesPath, index_col=0, delimiter=';') # All data in selectionFeatures.csv
    X = X[selectedFeatures] # Filter on the selected features
    idX = X.index.values # Patients with input data

    # Read output data from csv-files
    y = pd.read_csv(manualFeaturesPath, index_col=0, delimiter=';') # All data in manualFeatures.csv
    y = y[y['outcome'] >= 0] # Keep only patients with given outcome
    y = y[['outcome']] # Keep only outcome
    idY = y.index.values # Patients with output data

    # Select patiets that have both input and output
    patIds = np.array([id for id in idX if id in idY])
    X = X.loc[patIds]
    y = y.loc[patIds]

    # Divide data into train- and test-data
    testIds = [1, 8, 13, 20, 40, 44, 49, 55]
    trainIds = [v for v in X.index.values if v not in testIds]

    yTest = y.loc[testIds]
    Xtest = X.loc[testIds]
    yTrain = y.loc[trainIds]
    Xtrain = X.loc[trainIds]

    # Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if optimizeParams:
        # Convert all parameter settings to lists
        for k, v in params.items():
            if not isinstance(v, list):
                params[k] = [v]      

        # Find the best parameter setting
        params = search_model_params(Xtrain, yTrain, method, params, paramSearchResultsPath, scoringOptiMetric)

    else:
        # Convert all parameter settings to to single values. If list, first value is taken
        for k, v in params.items():
            if isinstance(v, list):
                params[k] = v[0]

    # Predict outcome of validation and test data, print some performance metrics
    yTrueVal, yPredRegVal = validate_model(Xtrain, yTrain, method, params)
    yTrueTest, yPredRegTest = test_model(Xtrain, Xtest, yTrain, yTest, method, params)

    return yTrueTest, yPredRegTest, yTrueVal, yPredRegVal, params

def search_model_params(Xtrain, yTrain, method, params, paramSearchResultsPath, scoringOptiMetric):
    """
    ACTION: 
        Computes the best parameter setting by searching the grid of parameter settings specified by the params-dictionary. 
    INPUTS: 
        Xtrain: DataFrame with training features
        yTrain: DataFrame with training labels
        method: machine learning algorithm to use, eg. 'RFreg' (random forest regression), 'RFclass' (random forest classifier)
        params: parameter settings for the selected method (a dictionary, values must be lists)
                For RFreg: n_estimators, max_features, max_depth
                For RFclass: n_estimators, max_features, max_depth
                For LogReg: penalty, solver, C, max_iter
        paramSearchResultsPath: path to paramSearchResults file
        scoringOptiMetric: metric to optimize over the given set of parameters
    OUTPUT:
        A dictionary with the best parameter setting
    """

    # Construct the ml model
    if method == 'RFreg':
        model = RandomForestRegressor(random_state=0)
    elif method == 'RFclass':
        model = RandomForestClassifier(random_state=0)
    elif method == 'LogReg':
        model = LogisticRegression(random_state=0)             
    else:
        print(f'Method "{method}" is not implemented in ml_prediction.py')
        return

    # Create model and do grid search
    modelSearch = GridSearchCV(model, params, scoring=scoringOptiMetric, cv = min(5, int(len(yTrain)/2)))
    modelSearch.fit(Xtrain.values, yTrain.values)

    # Create a csv-file with the results of the model-parameter-search
    df = pd.DataFrame(modelSearch.cv_results_)
    df.to_csv(paramSearchResultsPath, sep=';')

    # Return the best parameter setting
    return modelSearch.best_params_

def validate_model(Xtrain, yTrain, method, params):
    """
    ACTION: 
        K-fold cross validation on the training data and prints some validation results
    INPUTS: 
        Xtrain: DataFrame with training features
        yTrain: DataFrame with training labels
        method: machine learning algorithm to use, eg. 'RFreg' (random forest regression), 'RFclass' (random forest classifier)
        params: parameter settings for the selected method (a dictionary, values cannot be lists)
    OUTPUTS: 
        yTrue: Numpy-array with true outcome values
        yPredReg: Numpy-array with predicted regression outcome values
    """

    # Construct the ml model
    if method == 'RFreg':
        model = RandomForestRegressor(**params, random_state=0)
    elif method == 'RFclass':
        model = RandomForestClassifier(**params, random_state=0)
    elif method == 'LogReg':
        model = LogisticRegression(**params, random_state=0)
        Xtrain=(Xtrain-Xtrain.mean())/Xtrain.std() # Standardize data        
    else:
        print(f'Method "{method}" is not implemented in ml_prediction.py')
        return

    # Create k-fold object
    nSplits = 5
    kf = KFold(n_splits=nSplits, shuffle=True, random_state=15)
    
    # Init vectors for prediction values
    yPredReg = np.zeros(len(yTrain.index))
    yTrue = np.zeros(len(yTrain.index))
    
    for trainIndex, testIndex in kf.split(Xtrain):

        # Split into train and test data
        X1 = Xtrain.values[trainIndex]
        y1 = yTrain.values[trainIndex]
        X2 = Xtrain.values[testIndex]
        y2 = yTrain.values[testIndex]    

        # Train model and make prediction on the test data
        model.fit(X1, y1.ravel())
        yPredReg[testIndex] = model.predict(X2)
        yTrue[testIndex] = y2.ravel()

    # Print performance metrics, return true outcome and predicted values
    print_metrics(yTrue, yPredReg)
    return yTrue, yPredReg

def test_model(Xtrain, Xtest, yTrain, yTest, method, params):
    """
    ACTION: 
        Train a model on the train data with given parameters and evaluate it on the test data.  
    INPUTS: 
        Xtrain: DataFrame with training features
        Xtest: DataFrame with test features
        yTrain: DataFrame with training labels
        yTest: DataFrame with test labels
        method: machine learning algorithm to use, eg. 'RFreg' (random forest regression), 'RFclass' (random forest classifier)
        params: parameter settings for the selected method (a dictionary, values cannot be lists)
    OUTPUTS: 
        yTrue: Numpy-array with true outcome values
        yPredReg: Numpy-array with predicted regression outcome values
    """
    
    # Construct the ml model
    if method == 'RFreg':
        model = RandomForestRegressor(**params, random_state=0)
    elif method == 'RFclass':
        model = RandomForestClassifier(**params, random_state=0)
    elif method == 'LogReg':
        model = LogisticRegression(**params, random_state=0)
        # Standardize data: 
        Xtrain=(Xtrain-Xtrain.mean())/Xtrain.std()        
        Xtest=(Xtest-Xtrain.mean())/Xtrain.std()             
    else:
        print(f'Method "{method}" is not implemented in ml_prediction.py')
        return
    
    # Train model and make prediction on the test data
    model.fit(Xtrain, yTrain)
    yPredReg = model.predict(Xtest)
    yTrue = yTest.values.ravel()

    # Print performance metrics, return true outcome and predicted values
    print_metrics(yTrue, yPredReg)
    return yTrue, yPredReg


def print_metrics(yTrue, yPredReg):
    """
    docstring
    """
    yPredClass = np.round(yPredReg).astype(int)
    
    print('')
    print('Accuracy:          ', metrics.accuracy_score(yTrue, yPredClass))
    print('Precicion (micro): ', metrics.precision_score(yTrue, yPredClass, average='micro'))
    print('Recall (micro):    ', metrics.recall_score(yTrue, yPredClass, average='micro'))
    print('Precicion (macro): ', metrics.precision_score(yTrue, yPredClass, average='macro'))
    print('Recall (macro):    ', metrics.recall_score(yTrue, yPredClass, average='macro'))
    # print('AUC (macro):       ', metrics.roc_auc_score(yTrue, yPredReg, average='macro'))
    # print('AUC (weighted):    ', metrics.roc_auc_score(yTrue, yPredReg, average='weighted'))
    
def write_results_to_csv(predResultsPath, FSmethod, FSparams, selectedFeatures, MLmethod, MLparams, yTrueTest, yPredRegTest, yTrueVal, yPredRegVal):
    """
    docstring
    """
    pass


if __name__ == '__main__':
    yTrue = [1, 2, 2, 3, 1, 3, 2, 2]
    yPredReg = [2.1, 2.1, 1.65, 1.87, 1.45, 2.67, 2.78, 2.34]
    print_metrics(yTrue, yPredReg)
