import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def create_evaluate_model(method, params, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams, scoringOptiMetric = 'r2'):
    """
    ACTION: 
        Computes the best parameter setting (if optimizeParams is True) by searching the grid of parameter settings specified by the params-dictionary. 
        Train a model on the train data with given parameters and evaluate it on the test data.  
    INPUTS: 
        method: machine learning algorithm to use, eg. 'RF' (random forest)
        params: parameter settings for the selected method (a dictionary, values can be lists)
                For RF: n_estimators, max_features
        selectedFeatures: list of features to use when traning a model
        selectionFeaturesPath: path to selectionFeatures file
        manualFeaturesPath: path to manualFeatures file
        paramSearchResultsPath: path to paramSearchResults file
        optimizeParams: boolean, if True, GridSearchCV is used to find the best parameter setting
        scoringOptiMetric: metric to optimize over the given set of parameters
    """

    # Read data from csv-files
    X = pd.read_csv(selectionFeaturesPath, index_col=0, delimiter=';') # All data in selectionFeatures.csv
    X_diagnostics = [col for col in X if col.startswith('diagnostics')]
    X = X.drop(columns=X_diagnostics) # Data in selectionFeatures.csv, excluding diagnostic features
    X = X[selectedFeatures] # Filter on the selected features

    y = pd.read_csv(manualFeaturesPath, index_col=0, delimiter=';') # All data in manualFeatures.csv
    X = X[y.outcome != 0]
    y = y['outcome'] # Keep only outcome
    y = y.loc[X.index] # Select only output data for the patients for wich we have input data

    # Divide data into train- and test-data
    Xtrain, Xtest, yTrain, yTest = train_test_split(X,y, test_size=0.2, random_state=0, stratify=y)

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

    evaluate_model(Xtrain, Xtest, yTrain, yTest, method, params)

def search_model_params(Xtrain, yTrain, method, params, paramSearchResultsPath, scoringOptiMetric):
    """
    ACTION: 
        Computes the best parameter setting by searching the grid of parameter settings specified by the params-dictionary. 
    INPUTS: 
        Xtrain: DataFrame with training features
        yTrain: DataFrame with training labels
        method: machine learning algorithm to use, eg. 'RF' (random forest)
        params: parameter settings for the selected method (a dictionary, values must be lists)
                For RF: nTrees, maxFeatures
        paramSearchResultsPath: path to paramSearchResults file
        scoringOptiMetric: metric to optimize over the given set of parameters
    OUTPUT:
        A dictionary with the best parameter setting
    """

    # Construct the ml model
    if method == 'RF':
        regModel = RandomForestRegressor(random_state=0)
    else:
        print(f'Method "{method}" is not implemented in feature_selection.py')
        return

    # Create model and do grid search
    modelSearch = GridSearchCV(regModel, params, scoring=scoringOptiMetric, cv = min(5, int(len(yTrain)/2)))
    modelSearch.fit(Xtrain.values, yTrain.values)

    # Create a csv-file with the results of the model-parameter-search
    df = pd.DataFrame(modelSearch.cv_results_)
    df.to_csv(paramSearchResultsPath, sep=';')

    # Return the best parameter setting
    return modelSearch.best_params_

def evaluate_model(Xtrain, Xtest, yTrain, yTest, method, params):
    """
    ACTION: 
        Train a model on the train data with given parameters and evaluate it on the test data.  
    INPUTS: 
        Xtrain: DataFrame with training features
        Xtest: DataFrame with test features
        yTrain: DataFrame with training labels
        yTest: DataFrame with test labels
        method: machine learning algorithm to use, eg. 'RF' (random forest)
        params: parameter settings for the selected method (a dictionary, values cannot be lists)
    """
    
    # Construct the ml model
    if method == 'RF':
        # Extracting parameter settings for Random Forest
        regModel = RandomForestRegressor(**params, random_state=0)
    else:
        print(f'Method "{method}" is not implemented in feature_selection.py')
        return
    
    # Train model and make prediction on the test data
    regModel.fit(Xtrain, yTrain)
    yPred = regModel.predict(Xtest)
    yPredClass = np.round(yPred).astype(int)

    # Print a dataFrame with the test labels and the predicted ones
    df = pd.DataFrame(yTest)
    df.insert(1, 'predict', yPred, True)
    df.insert(2, 'predictClass', yPredClass, True)
    print(df)

    # Print regression metrics
    print('')
    print('Root Mean Square error: ', np.sqrt(metrics.mean_squared_error(yTest.values, yPred)))
    print('Mean Square error:      ', metrics.mean_squared_error(yTest.values, yPred))
    print('Mean Absolute error:    ', metrics.mean_absolute_error(yTest.values, yPred))
    print('R2-score:               ', metrics.r2_score(yTest.values, yPred))

    # Print classification metrics
    print('')
    print('Accuracy:  ', metrics.accuracy_score(yTest.values, yPredClass))
    print('Precicion: ', metrics.precision_score(yTest.values, yPredClass, average='micro'))
    print('Recall:    ', metrics.recall_score(yTest.values, yPredClass, average='micro'))
