# import image_preprocessing as imgpr
# import feature_extraction as fextr
# import feature_selection as fesel
import ml_prediction as mlpred
import time

# Settings for feature extraction
paramsPath = "Params.yaml"

dataPath = "../../patient_data"
selectionFeaturesPath = "../../patient_data/selection_features.csv"
manualFeaturesPath = "../../patient_data/manual_features.csv"
paramSearchResultsPath = "../../patient_data/param_search_results.csv"

img2use = ["T2"]
mask2use = ["M"]

# MRMR Feature selection
# mrmrParams = {'nFeatures': 10, 
#             'internalFEMethod': 'MID', 
#             'nBins': 3, 
#             'discStrategy': 'kmeans'}
# selectedFeatures = fesel.select_features('MRMR', mrmrParams, selectionFeaturesPath, manualFeaturesPath)
selectedFeatures = ['original_gldm_LargeDependenceLowGrayLevelEmphasis_T2_M', 'original_glcm_Imc2_T2_M', 'Patients Weight', 'original_firstorder_Minimum_T2_M', 'original_glcm_Id_T2_M', 'original_shape_SurfaceVolumeRatio_T2_M', 'original_glszm_LargeAreaLowGrayLevelEmphasis_T2_M', 'original_firstorder_Kurtosis_T2_M', 'original_glcm_Idm_T2_M', 'original_glcm_MCC_T2_M']

# Settings for random forest and parameter search
method = 'RF'
rfParams = {'n_estimators': [5, 10, 15], 
            'max_depth': [None, 1, 3, 5],
            'max_features': [0.33, 0.67, 1.0]}
scoring = 'r2' #'neg_root_mean_squared_error'

t0 = time.time()
mlpred.create_evaluate_model(method, rfParams, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams=False, scoringOptiMetric=scoring)
t1 = time.time()


# print('Elapsed time:', t1-t0)
