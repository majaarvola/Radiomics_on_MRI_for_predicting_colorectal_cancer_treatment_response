import image_preprocessing as imgpr
import feature_extraction as fextr
import feature_selection as fesel
import ml_prediction as mlpred
import time


dataPath = "../../patient_data"
selectionFeaturesPath = "../../patient_data/selection_features.csv"
manualFeaturesPath = "../../patient_data/manual_features.csv"
paramSearchResultsPath = "../../patient_data/param_search_results.csv"


t0 = time.time() # Start the timer

# # Image preprocessing
# imgpr.create_masks_and_nrrds(dataPath)

# t1 = time.time() # Lap the timer

# Feature extraction
img2use = ["T2"]
mask2use = ["M"]
paramsPath = "Params.yaml"
fextr.extract_features_from_all(dataPath, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesPath)

t2 = time.time() # Lap the timer

# # Feature selection
# method = 'LogReg' # 'MRMR' 'LASSO'
# mrmrParams = {'nFeatures': 20, 
#             'internalFEMethod': 'MID', 
#             'nBins': 3, 
#             'discStrategy': 'kmeans'}
# lassoParams = {'nFeatures': 20}
# selectedFeatures = fesel.select_features(method, lassoParams, selectionFeaturesPath, manualFeaturesPath)
# print(selectedFeatures)
# # selectedFeatures = ['original_gldm_LargeDependenceLowGrayLevelEmphasis_T2_M', 'original_glcm_Imc2_T2_M', 'Patients Weight', 'original_firstorder_Minimum_T2_M', 'original_glcm_Id_T2_M', 'original_shape_SurfaceVolumeRatio_T2_M', 'original_glszm_LargeAreaLowGrayLevelEmphasis_T2_M', 'original_firstorder_Kurtosis_T2_M', 'original_glcm_Idm_T2_M', 'original_glcm_MCC_T2_M']
# # selectedFeatures = ['log-sigma-1-mm-3D_glszm_SmallAreaEmphasis_T2_M', 'wavelet-HL_glcm_SumAverage_T2_M', 'squareroot_firstorder_RobustMeanAbsoluteDeviation_T2_M', 'wavelet-LH_ngtdm_Strength_T2_M', 'gradient_firstorder_Kurtosis_T2_M', 'wavelet-HH_firstorder_Skewness_T2_M', 'log-sigma-5-mm-3D_glcm_Imc2_T2_M', 'logarithm_glcm_ClusterProminence_T2_M', 'log-sigma-5-mm-3D_gldm_DependenceVariance_T2_M', 'squareroot_glcm_MCC_T2_M', 'original_gldm_LargeDependenceLowGrayLevelEmphasis_T2_M', 'log-sigma-5-mm-3D_glcm_DifferenceAverage_T2_M', 'wavelet-LH_glcm_ClusterProminence_T2_M', 'log-sigma-1-mm-3D_glcm_MCC_T2_M', 'log-sigma-5-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_M', 'original_glrlm_LongRunLowGrayLevelEmphasis_T2_M', 'wavelet-LH_glcm_DifferenceVariance_T2_M', 'wavelet-LH_glcm_Correlation_T2_M', 'wavelet-HL_gldm_HighGrayLevelEmphasis_T2_M', 'log-sigma-3-mm-3D_firstorder_10Percentile_T2_M']
# selectedFeatures = ['log-sigma-3-mm-3D_firstorder_10Percentile_T2_M', 'wavelet-LH_glcm_InverseVariance_T2_M', 'square_glszm_LowGrayLevelZoneEmphasis_T2_M', 'wavelet-HL_ngtdm_Strength_T2_M', 'logarithm_glcm_ClusterProminence_T2_M', 'log-sigma-1-mm-3D_glcm_Correlation_T2_M', 'square_glcm_Idn_T2_M', 'lbp-2D_glrlm_HighGrayLevelRunEmphasis_T2_M', 'wavelet-LH_firstorder_Skewness_T2_M', 'wavelet-LL_glszm_LargeAreaLowGrayLevelEmphasis_T2_M', 'gradient_glcm_SumSquares_T2_M', 'wavelet-LL_firstorder_Energy_T2_M', 'lbp-2D_glrlm_LowGrayLevelRunEmphasis_T2_M', 'squareroot_glcm_Imc2_T2_M', 'squareroot_firstorder_Minimum_T2_M', 'log-sigma-1-mm-3D_firstorder_RobustMeanAbsoluteDeviation_T2_M', 'wavelet-LH_ngtdm_Busyness_T2_M', 'log-sigma-5-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis_T2_M', 'squareroot_firstorder_InterquartileRange_T2_M', 'original_gldm_LargeDependenceLowGrayLevelEmphasis_T2_M']

t3 = time.time() # Lap the timer

# # Prediction model
# method = 'RFclass' #'RF'
# rfParams = {'n_estimators': [5, 10, 15, 20], 
#             'max_depth': [None, 1, 3, 5],
#             'max_features': [0.33, 0.67, 1.0, 'auto']}
# scoring = 'accuracy' #'r2' 'neg_root_mean_squared_error'
# mlpred.create_evaluate_model(method, rfParams, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams=True, scoringOptiMetric=scoring)

t4 = time.time() # Stop the timer

# print('Elapsed time, Image preprocessing:', t1-t0)
# print('Elapsed time, Feature extraction :', t2-t1)
# print('Elapsed time, Feature selection  :', t3-t2)
# print('Elapsed time, Prediction model   :', t4-t3)
print('Elapsed total time:', t4-t0)
