import sys
sys.path.append('../')

import image_processing as imgpr
import feature_extraction as fextr
import feature_selection as fesel
import ml_prediction as mlpred
import time


dataPath = "../../../patient_data"
selectionFeaturesPath = "../../../patient_data/selection_features_M+.csv"
manualFeaturesPath = "../../../patient_data/manual_features_class.csv"
paramSearchResultsPath = "../../../patient_data/param_search_results.csv"
predResultsPath = "../../../patient_data/prediction_results.csv"

t0 = time.time() # Start the timer

# Image preprocessing
imgpr.create_masks_and_nrrds(dataPath)

t1 = time.time() # Lap the timer

# # Feature extraction
# img2use = ["T2"]
# mask2use = ["M"]
# paramsPath = "Params.yaml"
# fextr.extract_features_from_all(dataPath, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesPath)

t2 = time.time() # Lap the timer

# # Feature selection
# FSmethod = 'MRMR'
# FSparams = {'nFeatures': 15, 
#             'internalFEMethod': 'MID', 
#             'nBins': 4, 
#             'discStrategy': 'kmeans'}
# selectedFeatures = fesel.select_features(FSmethod, FSparams, selectionFeaturesPath, manualFeaturesPath)
# print('selectedFeatures =', selectedFeatures)

# FSmethod = 'LogReg'
# FSparams = {'nFeatures': 15}
# selectedFeatures = ['lbp-3D-m1_firstorder_Range_T2_M+', 'lbp-2D_glcm_JointEntropy_T2_M+', 'log-sigma-5-mm-3D_firstorder_90Percentile_T2_M+', 'square_glcm_Imc1_T2_M+', 'wavelet-LH_gldm_DependenceVariance_T2_M+', 'exponential_glcm_MCC_T2_M+', 'lbp-3D-m2_firstorder_Median_T2_M+', 'wavelet-HL_glszm_ZonePercentage_T2_M+', 'lbp-3D-m1_firstorder_Skewness_T2_M+', 'log-sigma-5-mm-3D_glszm_SmallAreaLowGrayLevelEmphasis_T2_M+', 'log-sigma-1-mm-3D_glrlm_LongRunHighGrayLevelEmphasis_T2_M+', 'log-sigma-1-mm-3D_glszm_SmallAreaEmphasis_T2_M+', 'logarithm_ngtdm_Contrast_T2_M+', 'wavelet-HH_glcm_SumSquares_T2_M+', 'log-sigma-3-mm-3D_firstorder_Skewness_T2_M+']

# FSmethod = 'LASSO'
# FSparams = {'nFeatures': 15}
# selectedFeatures = ['square_glcm_Idmn_T2_M+', 'log-sigma-5-mm-3D_glszm_SmallAreaLowGrayLevelEmphasis_T2_M+', 'wavelet-LH_glrlm_ShortRunLowGrayLevelEmphasis_T2_M+', 'lbp-2D_glrlm_ShortRunEmphasis_T2_M+', 'logarithm_ngtdm_Contrast_T2_M+', 'wavelet-LL_glcm_InverseVariance_T2_M+', 'logarithm_gldm_DependenceNonUniformityNormalized_T2_M+', 'log-sigma-5-mm-3D_ngtdm_Contrast_T2_M+', 'wavelet-HH_firstorder_Skewness_T2_M+', 'log-sigma-3-mm-3D_glszm_GrayLevelVariance_T2_M+', 'wavelet-HL_firstorder_Skewness_T2_M+', 'log-sigma-5-mm-3D_firstorder_90Percentile_T2_M+', 'lbp-3D-m1_glrlm_ShortRunEmphasis_T2_M+', 'log-sigma-3-mm-3D_firstorder_Median_T2_M+', 'squareroot_glrlm_LongRunHighGrayLevelEmphasis_T2_M+']

# FSmethod = 'MRMR'
# FSparams = {'nFeatures': 15, 'internalFEMethod': 'MID', 'nBins': 4, 'discStrategy': 'kmeans'}
# selectedFeatures = ['wavelet-LL_glcm_Idn_T2_M+', 'log-sigma-5-mm-3D_firstorder_90Percentile_T2_M+', 'square_firstorder_InterquartileRange_T2_M+', 'wavelet-HL_glrlm_LongRunLowGrayLevelEmphasis_T2_M+', 'wavelet-HH_glcm_ClusterShade_T2_M+', 'log-sigma-1-mm-3D_firstorder_Kurtosis_T2_M+', 'wavelet-LH_glszm_SmallAreaEmphasis_T2_M+', 'square_glcm_Correlation_T2_M+', 'lbp-3D-m1_firstorder_MeanAbsoluteDeviation_T2_M+', 'wavelet-HL_glcm_DifferenceVariance_T2_M+', 'wavelet-HH_glszm_ZonePercentage_T2_M+', 'lbp-3D-m1_firstorder_Skewness_T2_M+', 'logarithm_ngtdm_Strength_T2_M+', 'wavelet-LH_glcm_SumSquares_T2_M+', 'lbp-3D-k_glrlm_ShortRunLowGrayLevelEmphasis_T2_M+']

t3 = time.time() # Lap the timer

# # Prediction model
# MLmethod = 'LogReg'
# # rfParams = {'n_estimators': [5, 10, 15, 25, 50, 75], 
# #             'max_depth': [None, 1, 3, 5, 10, 15],
# #             'max_features': [0.33, 0.67, 1.0, 'sqrt']}
# logRegParams = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
#             'penalty': 'l1',
#             'solver': 'liblinear', 
#             'max_iter': 2000}
            
# scoring = 'accuracy' #'neg_root_mean_squared_error' 'r2'
# yTrueTest, yPredRegTest, yTrueVal, yPredRegVal, MLparams = mlpred.create_evaluate_model\
#     (MLmethod, logRegParams, selectedFeatures, selectionFeaturesPath, manualFeaturesPath, paramSearchResultsPath, optimizeParams=True, scoringOptiMetric=scoring)

t4 = time.time() # Stop the timer

# mlpred.write_results_to_csv(predResultsPath, selectionFeaturesPath, FSmethod, FSparams, selectedFeatures, MLmethod, MLparams, yTrueTest, yPredRegTest, yTrueVal, yPredRegVal)

# print('')
# print('Elapsed time, Image preprocessing:', t1-t0)
# print('Elapsed time, Feature extraction :', t2-t1)
# print('Elapsed time, Feature selection  :', t3-t2)
# print('Elapsed time, Prediction model   :', t4-t3)
# print('Elapsed total time:', t4-t0)
