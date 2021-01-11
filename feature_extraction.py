import image_processing as imgpr
import cv2 
import os
from radiomics import featureextractor, setVerbosity
import scipy
import trimesh
import csv
import re 

def extract_manual_feature(manualFeaturesPath, features):
    """
    ACTION: 
        Open a csv-file with patient information created manually, read the content and create a dictionary with dictionaries containing specified features
    INPUT: 
        manualFeaturesPath: Path to csv-file containing patient numbers and manually calculated features, structure of the file is specified in 'README.md'
        features: A list of features (strings) to put in the dictionary 
    OUTPUT: 
        allPatsDict: A dictionary with patient number (string) as key, and dictionary with features as value
    """
    allPatsDict = {} # Directory containing directories
    with open(manualFeaturesPath, 'r', newline='') as csvFile: # Open the file
        reader = csv.DictReader(csvFile, delimiter=';') # Create a reader
        for row in reader: # For every row in the csv-file, add a tuple to the dictionary
            patDict = {} # Create a directory with features for every patient
            for feature in features:
                patDict[feature] = row[feature] # Adding feature to patients directory
            allPatsDict[row['id']] = patDict # Adding the directory to the 'One directory to rule them all'
    return allPatsDict

def extract_features_from_image(imagePath, maskPath, paramsPath):
    """
    ACTION: 
        Extract radiomic features from one image, given mask and parameters, 
        and return a dictionary with the feature values
    INPUT: 
        imagePath: path to image nrrd-file (without extension)
        maskPath: path to mask nrrd-file (without extension)
        paramsPath: path to file with radiomic feature extraction settings
    OUTPUT: 
        dictionary with all extracted features
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
    setVerbosity(40) # Set level of verbosity, 40=Errors, 30=Warnings, 20=Info, 10=Debug
    results = extractor.execute(img, mask)
    return results


def extract_features_from_patient(dataPath, patientId, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesDict, dicomFeaturesDict):
    """
    ACTION: 
        Extract all features from a patient and write the result to a csv (selectionFeaturesPath).
    INPUT:  
        dataPath: path to the folder containg all the data, the structure inside this folder is important and specified in 'README.md'
        patientId: Patient ID in string format, ex "13"
        img2use: list of types of scans to extract features from, ex ["T1", "T2", "Diff", "ADC"]
        mask2use: list of masks to use for extracting features, ex ["M", "M+", "Mfrisk"]
        paramsPath: path to file with radiomic feature extraction settings
        selectionFeaturesPath: path to csv-file where to write the result
        manualFeaturesDict: Dictionary with manually calculated features for this patient
        dicomFeaturesDict: Dictionary with features for this patient extracted from the html-file with DICOM header information
    OUTPUT: 
        1 if extraction was successful, 0 if not
    """    

    # Add patient ID first in the dictionary
    features = {"patientId": patientId} 

    # Add manually extracted, and dicom- features to the directory 
    features.update(manualFeaturesDict)
    features.update(dicomFeaturesDict)

    # Create path to patient folder
    patientFolder = dataPath + '/' + 'Pat' + patientId + '/'

    # Go through all images to use and create paths for each
    for img in img2use:
        imagePath = patientFolder + 'Pat' + patientId + img + 'U'

        # Go through all masks and create mask paths
        for mask in mask2use:
            maskPath = patientFolder + 'Pat' + patientId + img + mask + '_mask'

            # Extract radiomic features
            try: 
                radiomicFeaturesDict = extract_features_from_image(imagePath, maskPath, paramsPath)
            except:
                return 0 # Extraction failed

            # Add suffix specifying image and mask and append to features dictionary
            radiomicFeaturesDict = {k + '_' + img + '_' + mask: v for k, v in radiomicFeaturesDict.items()}
            features.update(radiomicFeaturesDict)

    # List all the feature names  
    header = list(features.keys())  
    
    # If file is empty we create the header and then add the content
    filesize = os.path.getsize(selectionFeaturesPath)
    if filesize == 0:
        with open(selectionFeaturesPath, 'w', newline='') as featuresFile:
            writer = csv.DictWriter(featuresFile, fieldnames=header, delimiter = ';')
            writer.writeheader()
            writer.writerow(features)

    # If file is not empty we append the new content
    else:
        with open(selectionFeaturesPath, 'a', newline='') as featuresFile:
            writer = csv.DictWriter(featuresFile, fieldnames=header, delimiter = ';')
            writer.writerow(features)
    
    return 1 # Successful extraction


def extract_features_from_all(dataPath, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesPath):
    """
    ACTION: 
        Loops through all patient folders, for the patiets that we have all images, manual features, and html-file with DICOM header information. 
        Extract features and store the content in a csv-file
    INPUT:  
        dataPath: path to the folder containg all the data, the structure inside this folder is important and specified in 'README.md'
        img2use: list of types of scans to extract features from, ex ["T1", "T2", "Diff", "ADC"]
        mask2use: list of masks to use for extracting features, ex ["M", "M+", "Mfrisk"]
        paramsPath: path to file with radiomic feature extraction settings
        selectionFeaturesPath: path to csv-file where to write all the extracted features
        manualFeaturesPath: Path to csv-file containing patient numbers and manually calculated features
    """

    with open(selectionFeaturesPath, "w") as f:
        f.truncate()    

    # Collect
    manualFeaturesDict = extract_manual_feature(manualFeaturesPath, ['age', 'treatment'])
    dicomFeaturesDict = imgpr.extract_dicom_features(dataPath, ['Patients Sex', 'Patients Weight'])

    # All patient ID:s with a folder with images
    folderContent = os.listdir(dataPath)
    imageIds = [x[3:] for x in folderContent if re.search('^Pat[0-9]?[0-9]?[0-9]$', x)]

    # All patient ID:s with a DICOM-file
    dicomIds = list(dicomFeaturesDict.keys())

    # All patient ID:s with given age and outcome
    manualIdsDict = extract_manual_feature(manualFeaturesPath, ['age', 'outcome', 'treatment'])
    manualIds = []
    for patId, patDict in manualIdsDict.items():
        if float(patDict['age'])>0 and float(patDict['outcome']) >= 0 and float(patDict['treatment']) >= 0:
            manualIds.append(patId)

    # All patient ID:s with images, dicom-files, age and outcome
    patIds = []
    for id in imageIds:
        if id in dicomIds: # and id in manualIds:
            patIds.append(id)

    # Extract features for every patient and put the result in a file
    for patientId in sorted(patIds, key=float):
        if extract_features_from_patient(dataPath, patientId, img2use, mask2use, paramsPath, selectionFeaturesPath, manualFeaturesDict[patientId], dicomFeaturesDict[patientId]):
            print(f"Pat{patientId}: features extracted")
        else:
            print(f"Pat{patientId}: failed to extract features")
