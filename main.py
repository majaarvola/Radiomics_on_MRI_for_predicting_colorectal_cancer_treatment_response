import image_preprocessing as imgpr
import feature_extraction as fextr
# import nrrd
# import cv2


# Extract label data from patients (either here or in seperate file)

# outputFile = "../../patient_data/10T2M_mask.tiff"
# inputFileNRRD = "../../patient_data/10T2M.nrrd"
# outputFileNRRD = "../../patient_data/10T2M_mask.nrrd"


# if cv2.haveImageReader(outputFile):
#     original_image = cv2.imread(outputFile)
#     nrrd.write(outputFileNRRD, original_image)

# showResult = True
# # imgpr.create_mask(inputFile, outputFile, showResult)

inputFile = "../../patient_data/10T2M"
paramsPath = "Params.yaml"

# imgpr.img2nrrd(inputFile)

features = fextr.extract_features_from_image(inputFile, paramsPath)
fextr.print_features(features)
