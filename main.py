import image_preprocessing as imgpr
import feature_extraction as fextr

# Extract label data from patients (either here or in seperate file)


# Settings for feature extraction
paramsPath = "Params.yaml"

# dirPath = "../../patient_data"
# imgpr.create_masks_and_nrrds(dirPath, overWrite=False)

dirMaskPath = "../../patient_data/Pat14/Pat14T2M_mask"
# imgpr.create_3d_nrrd(dirMaskPath)
dirImgPath = "../../patient_data/Pat14/Pat14T2U"
# imgpr.create_3d_nrrd(dirImgPath)

results = fextr.extract_features_from_image(dirImgPath, dirMaskPath, paramsPath) 
fextr.print_features(results)
