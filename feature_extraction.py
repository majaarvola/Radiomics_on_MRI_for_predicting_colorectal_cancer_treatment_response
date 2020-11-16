import cv2 
import os
from radiomics import featureextractor
import scipy
import trimesh
import csv

def extract_features_from_image(imagePath, maskPath, paramsPath):
    """
    ACTION: Extract radiomic features from one image given mask and parameters
    INPUT: imagePath, maskPath, paramsPath
    OUTPUT: dictionary
    """

    img = imagePath + ".nrrd"
    mask = maskPath + ".nrrd"

    if not os.path.isfile(img):
        raise IOError('File does not exist: %s' % img)
    elif not os.path.isfile(mask):
        raise IOError('File does not exist: %s' % mask)
    elif not os.path.isfile(paramsPath):
        raise IOError('File does not exist: %s' % paramsPath)

    extractor = featureextractor.RadiomicsFeatureExtractor(paramsPath)
    results = extractor.execute(img, mask)
    return results


def extract_features_from_patient(patientId, img2use, mask2use, paramsPath, featuresPath):
    """
    ACTION: Extract all features from a patient and write the result in the featuresFile.
    INPUT:  patientId
            img2use: list of types of scans to extract features from, ex ["T1", "T2", "Diff", "ADC"]
            mask2use: list of masks to use for extracting features, ex ["M", "M+", "Mfrisk"]
            paramsPath: parameter path
            featuresPath: path to file where to write the result.
    """    






def extract_features_from_all(folderPath, img2use, mask2use, paramsPath, featuresPath):
    """
    ACTION: Loops through all patients, extract features and store the content in a table
    INPUT:  folderPath
            img2use: list of types of scans to extract features from, ex ["T1", "T2", "Diff", "ADC"]
            mask2use: list of masks to use for extracting features, ex ["M", "M+", "Mfrisk"]
            paramsPath: parameter path
            featuresPath: path to file where to write all the extracted features
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
