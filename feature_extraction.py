import cv2 
import os
from radiomics import featureextractor
import scipy
import trimesh

def extract_features_from_image(imagePath, maskPath, paramsPath):
    """
    ACTION: Extract radiomic features from one image given mask and parameters
    INPUT: imagePath, maskPath, paramsPath
    OUTPUT: dictionary
    """

    # img = imagePath + ".nrrd"
    # mask = imagePath + "_mask.nrrd"
    img = imagePath + ".nrrd"
    mask = maskPath + ".nrrd"
    print(img)
    print(mask)

    if not os.path.isfile(img):
        raise IOError('File does not exist: %s' % img)
    elif not os.path.isfile(mask):
        raise IOError('File does not exist: %s' % mask)
    elif not os.path.isfile(paramsPath):
        raise IOError('File does not exist: %s' % paramsPath)

    extractor = featureextractor.RadiomicsFeatureExtractor(paramsPath)
    results = extractor.execute(img, mask)
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
