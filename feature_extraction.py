import cv2 
import os
from radiomics import featureextractor

def extract_features_from_image(imagePath, paramsPath):
    """
    ACTION: Extract radiomic features from one image given mask and parameters
    INPUT: imagePath, paramsPath
    OUTPUT: dictionary
    """

    img = imagePath + ".nrrd"
    mask = imagePath + "_mask.nrrd"

    if not os.path.isfile(img):
        raise IOError('File does not exist: %s' % img)
    elif not os.path.isfile(mask):
        raise IOError('File does not exist: %s' % mask)
    elif not os.path.isfile(paramsPath):
        raise IOError('File does not exist: %s' % paramsPath)

    extractor = featureextractor.RadiomicsFeatureExtractor(paramsPath)
    results = extractor.execute(img, mask, label=255)
    return results


def extract_features_from_all(folderPath, paramsPath):
    """
    ACTION: Loops through all patients, extract features and store the content in a table
    INPUT: folder path and parameter path
    OUTPUT: table where each row corresponds to one patient and each column
            corresponds to one feature 
            (CHECK THAT THIS IS VALID INPUT FOR NEXT STEP IN PIPELINE)
    """
    pass

def print_features(features):
    """
    ACTION: Print the radiomics features
    INPUT: dictionary with radiomics features
    """
    print("Calculated features")
    for key, value in features.items():
        print("\t", key, ":", value)
