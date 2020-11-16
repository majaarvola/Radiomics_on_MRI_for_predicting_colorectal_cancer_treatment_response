import cv2 
import os
from radiomics import featureextractor
import scipy
import trimesh
import csv
import re 

def extract_features_from_image(imagePath, maskPath, paramsPath):
    """
    ACTION: 
        Extract radiomic features from one image given mask and parameters
    INPUT: 
        imagePath: path to image nrrd
        maskPath: path to mask nrrd
        paramsPath: path to file where to write the result
    OUTPUT: 
        dictionary
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


def extract_features_from_patient(dataPath, patientId, img2use, mask2use, paramsPath, featuresPath):
    """
    ACTION: 
        Extract all features from a patient and write the result in the featuresFile.
    INPUT:  
        patientId: Patient ID in string format, ex "13"
        img2use: list of types of scans to extract features from, ex ["T1", "T2", "Diff", "ADC"]
        mask2use: list of masks to use for extracting features, ex ["M", "M+", "Mfrisk"]
        paramsPath: parameter path
        featuresPath: path to file where to write the result
    """    
    #Add patient ID first in the dictionary
    features = {"patientId": patientId} 

    # Create path to patient folder
    patientFolder = dataPath + '/' + 'Pat' + patientId + '/'

    # Go through all images to use and create paths for each
    for img in img2use:
        imagePath = patientFolder + 'Pat' + patientId + img + 'U'

        # Go through all masks and create mask paths
        for mask in mask2use:
            maskPath = patientFolder + 'Pat' + patientId + 'T2' + mask + '_mask'

            # Extract features
            features_temp = extract_features_from_image(imagePath, maskPath, paramsPath)

            # Add suffix specifying image and mask and append to features dictionary
            features_temp = {k + '_' + img + '_' + mask: v for k, v in features_temp.items()}
            features.update(features_temp)

    # List all the feature names  
    header = list(features.keys())  
    
    # If file is empty we create the header and then add the content
    filesize = os.path.getsize(featuresPath)
    if filesize == 0:
        with open(featuresPath, 'w', newline='') as featuresFile:
            writer = csv.DictWriter(featuresFile, fieldnames=header, delimiter = ';')
            writer.writeheader()
            writer.writerow(features)

    # If file is not empty we append the new content
    else:
        with open(featuresPath, 'a', newline='') as featuresFile:
            writer = csv.DictWriter(featuresFile, fieldnames=header, delimiter = ';')
            writer.writerow(features)



def extract_features_from_all(dataPath, img2use, mask2use, paramsPath, featuresPath):
    """
    ACTION: 
        Loops through all patients, extract features and store the content in a file
    INPUT:  
        dataPath
        img2use: list of types of scans to extract features from, ex ["T1", "T2", "Diff", "ADC"]
        mask2use: list of masks to use for extracting features, ex ["M", "M+", "Mfrisk"]
        paramsPath: parameter path
        featuresPath: path to file where to write all the extracted features
    """

    with open(featuresPath, "w") as f:
        f.truncate()    

    # Create list with all existing patient IDs in the data folder
    folderContent = os.listdir(dataPath)
    patIds = [int(x[3:]) for x in folderContent if re.search('^Pat[0-9]?[0-9]?[0-9]$', x)]

    for patientId in sorted(patIds):
        extract_features_from_patient(dataPath, str(patientId), img2use, mask2use, paramsPath, featuresPath)
        print(f"Pat{patientId}: features extracted")


def print_features(features):
    """
    ACTION: 
        Print the radiomics features
    INPUT: 
        dictionary with radiomics features
    """
    print("Calculated features")
    for key, value in features.items():
        print("\t", key, ":", value)
