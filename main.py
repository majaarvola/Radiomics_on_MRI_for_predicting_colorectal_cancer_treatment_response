import image_preprocessing as imgpr
import feature_extraction as fextr
import time
# Extract label data from patients (either here or in seperate file)

# Settings for feature extraction
paramsPath = "Params.yaml"

dataPath = "../../patient_data"
featuresPath = "../../patient_data/selection_features.csv"
manualFeaturesPath = "../../patient_data/manual_features.csv"

img2use = ["T2"]
mask2use = ["M"]

t0 = time.time()

t1 = time.time()


print('Elapsed time', t1-t0)
