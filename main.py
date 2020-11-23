import image_preprocessing as imgpr
import feature_extraction as fextr
import time
# Extract label data from patients (either here or in seperate file)

# Settings for feature extraction
paramsPath = "Params.yaml"

dataPath = "../../patient_data"
featuresPath = "../../patient_data/selection_features.csv"

img2use = ["T2"]
mask2use = ["M"]

t0 = time.time()
# imgpr.erosion_manual_masks(dataPath)


imgpr.html_to_csv(dataPath)
# imgpr.create_masks_and_nrrds(dataPath, overWrite=False)
t1 = time.time()
# fextr.extract_features_from_all(dataPath, img2use, mask2use, paramsPath, featuresPath)
# t2 = time.time()

print('Elapsed time', t1-t0)
# print('Elapsed time extract features    :', t2-t1)

# dirMaskPath = "../../patient_data/Pat14/Pat14T2M_mask"
# imgpr.create_3d_nrrd(dirMaskPath)
# dirImgPath = "../../patient_data/Pat14/Pat14T2U"
# imgpr.create_3d_nrrd(dirImgPath)

# results = fextr.extract_features_from_image(dirImgPath, dirMaskPath, paramsPath) 
# fextr.print_features(results)
