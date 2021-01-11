import sys
sys.path.append('../')

import image_processing as imgpr
import feature_extraction as fextr
import feature_selection as fesel
import ml_prediction as mlpred


dataPath = "../../../patient_data"
selectionFeaturesPath = "../../../patient_data/selection_features.csv"
manualFeaturesPath = "../../../patient_data/manual_features_class.csv"
paramSearchResultsPath = "../../../patient_data/param_search_results.csv"
predResultsPath = "../../../patient_data/prediction_results.csv"


# Image processing
imgpr.create_masks_and_nrrds(dataPath)

# Feature extraction
img2use = ["T2"]
mask2use = ["M+"]
paramsPath = "Params.yaml"
fextr.extract_features_from_all(dataPath, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesPath)


# Feature selection
FSmethod = 'MRMR'
FSparams = {'nFeatures': 15, 
            'internalFEMethod': 'MID', 
            'nBins': 4, 
            'discStrategy': 'kmeans'}
selectedFeatures = fesel.select_features(FSmethod, FSparams, selectionFeaturesPath, manualFeaturesPath)
print(f'Features selected by {FSmethod}:')
print(selectedFeatures)


# Prediction model
MLmethod = 'RFreg'
rfParams = {'n_estimators': [5, 10, 15, 25, 50, 75], 
            'max_depth': [None, 1, 3, 5, 10, 15],
            'max_features': [0.33, 0.67, 1.0, 'sqrt']}  
scoring = 'r2'
yTrueTest, yPredRegTest, yTrueVal, yPredRegVal, MLparams = mlpred.create_evaluate_model\
    (MLmethod, rfParams, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams=True, scoringOptiMetric=scoring)

# Write validation- and test- results to csv-file
mlpred.write_results_to_csv(predResultsPath, selectionFeaturesPath, FSmethod, FSparams, selectedFeatures, MLmethod, MLparams, yTrueTest, yPredRegTest, yTrueVal, yPredRegVal)
