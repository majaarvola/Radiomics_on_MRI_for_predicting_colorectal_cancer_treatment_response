import image_preprocessing as imgpr
import feature_extraction as fextr

# Extract label data from patients (either here or in seperate file)


# # Settings for feature extraction
# # paramsPath = "Params.yaml"
# paramsPath = "MR_2D_extraction.yaml"

# # Given images
# inputFileNameM = "../../patient_data/10T2M" # With green line
# inputFileNameU = "../../patient_data/10T2U" # Without green line
# fileExt = ".tiff"

# # Create a mask
# maskFileName = "../../patient_data/10T2M_mask"
# showResult = False
# imgpr.create_mask(inputFileNameM + fileExt, maskFileName + fileExt, showResult)

# imgpr.img2nrrd(maskFileName)
# imgpr.img2nrrd(inputFileNameU)

# features = fextr.extract_features_from_image(inputFileNameU, maskFileName, paramsPath)

# fextr.print_features(features)
# print("Total number of features: ", len(features))


dirPath = "..\..\patient_data"
imgpr.create_masks_and_nrrds(dirPath, overWrite=True)
