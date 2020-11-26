# import image_preprocessing as imgpr
# import feature_extraction as fextr
import feature_selection as fesel
import time
# Extract label data from patients (either here or in seperate file)

# Settings for feature extraction
paramsPath = "Params.yaml"

dataPath = "../../patient_data"
selectionFeaturesPath = "../../patient_data/selection_features.csv"
manualFeaturesPath = "../../patient_data/manual_features.csv"

img2use = ["T2"]
mask2use = ["M"]

params = {'nFeatures': 10, 
            'internalFEMethod': 'MID', 
            'nBins': 3, 
            'discStrategy': 'kmeans'}

t0 = time.time()
selectedFeatures = fesel.select_features('MRMR', params, selectionFeaturesPath, manualFeaturesPath)
t1 = time.time()

print(selectedFeatures)

print('Elapsed time:', t1-t0)
