import image_preprocessing as imgpr
import feature_extraction as fextr
import feature_selection as fesel
import ml_prediction as mlpred
import time


dataPath = "../../patient_data"
selectionFeaturesPath = "../../patient_data/selection_features_M+.csv"
manualFeaturesPath = "../../patient_data/manual_features_class.csv"
paramSearchResultsPath = "../../patient_data/param_search_results.csv"
predResultsPath = "../../patient_data/prediction_results.csv"

t0 = time.time() # Start the timer

# # Image preprocessing
# imgpr.create_masks_and_nrrds(dataPath)

t1 = time.time() # Lap the timer

# # Feature extraction
# img2use = ["T2"]
# mask2use = ["M"]
# paramsPath = "Params.yaml"
# fextr.extract_features_from_all(dataPath, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesPath)

t2 = time.time() # Lap the timer

# # Feature selection
# FSmethod = 'LASSO'
# FSparams = {'nFeatures': 20, 
#             'internalFEMethod': 'MID', 
#             'nBins': 5, 
#             'discStrategy': 'kmeans'}
# selectedFeatures = fesel.select_features(FSmethod, FSparams, selectionFeaturesPath, manualFeaturesPath)
# print(selectedFeatures)

t3 = time.time() # Lap the timer

# # Prediction model
# MLmethod = 'RFreg'
# rfParams = {'n_estimators': [5, 10, 15, 25], 
#             'max_depth': [None, 1, 3, 5],
#             'max_features': [0.33, 0.67, 1.0, 'sqrt']}
# # logRegParams = {'C': [0.0001, 0.01, 1, 100, 10000], 
# #             'penalty': 'l1',
# #             'solver': 'liblinear', 
# #             'max_iter': 2000}
            
# scoring = 'r2' #'neg_root_mean_squared_error'
# yTrueTest, yPredRegTest, yTrueVal, yPredRegVal, MLparams = mlpred.create_evaluate_model\
#     (MLmethod, rfParams, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams=True, scoringOptiMetric=scoring)

t4 = time.time() # Stop the timer

# mlpred.write_results_to_csv(predResultsPath, selectionFeaturesPath, FSmethod, FSparams, selectedFeatures, MLmethod, MLparams, yTrueTest, yPredRegTest, yTrueVal, yPredRegVal)

print('')
print('Elapsed time, Image preprocessing:', t1-t0)
print('Elapsed time, Feature extraction :', t2-t1)
print('Elapsed time, Feature selection  :', t3-t2)
print('Elapsed time, Prediction model   :', t4-t3)
print('Elapsed total time:', t4-t0)
