
import numpy as np
import nrrd
import radiomics
import cv2
import os
import re
import csv

def create_mask(imagePath, maskPath, addInterior=True):
    """ 
    ACTION: 
        Generates a black and white mask of the input image
        The white area corresponds to green markings in the
        file including any interior points and the rest is black.
    INPUTS: 
        imagePath: path to image file
        maskPath: path of mask file to be created
    OUTPUT: 
        1 if mask was created, 0 if not
    """

    # Read image (if it exists) and make copy for comparison
    if cv2.haveImageReader(imagePath):
        originalImage = cv2.imread(imagePath)
    else:
        print(f"Failed to read input file at {imagePath}")
        return 0
    image = originalImage.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range and mask everything within range
    lower = np.array([50, 125, 125], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    if addInterior:
        # Add interior points to mask
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        mask = cv2.fillPoly(mask, cnts, (255, 255, 255))
        mask = erosion(mask) # Perform erosion on mask

    # save the output
    if not cv2.imwrite(maskPath, mask):
        print(f"Failed to write output file at {maskPath}")

    return 1

def erosion(mask):
    """
    ACTION: 
        Performs erosion on the input image, returns the result
    INPUTS: 
        mask: image to perform erosion on
    """
    kernel = np.ones((3,3),np.uint8)
    kernel[0,0] = 0
    kernel[0,-1] = 0
    kernel[-1,0] = 0
    kernel[-1,-1] = 0
    return cv2.erode(mask,kernel,iterations = 1)


def create_3d_nrrd(dataPath, patientDict):
    """
    ACTION: 
        Create three dimensional nrrd-file from all tiff-files in given folder, 
        the nrrd-file will be located next to the folder with the same name
    INPUTS: 
        dataPath: folder with tiff-files
        patientDict: dictionary with DICOM features for the patient
    OUTPUT: 
        1 if nrrd-file was created, 0 if not
    """

    if not os.path.isdir(dataPath): return 0 # Exit if folder does not exist

    fileNames = os.listdir(dataPath) # Read content in the folder
    tiffFileNames = [x for x in fileNames if x.endswith('.tiff')] # List with all tiff-images
    nrTiffFiles = len(tiffFileNames) # Number of tiff-images in folder

    if nrTiffFiles == 0: return 0 # Exit if folder contains no tiff-files

    firstImage = cv2.imread(dataPath + '\\' + tiffFileNames[0], 0)
    w, h = firstImage.shape # Width and height of the first tiff-image in the folder

    # Make 3D-nrrd-file from all tiff-files in the folder
    i = 0
    data = np.zeros((w,h,nrTiffFiles)) # Allocate 3D-array for data, 400x400 IMAGES ASSUMED
    for fileName in sorted(tiffFileNames):
        # Insert grayscaled layer into 3D-array
        data[:,:,i] = cv2.imread(dataPath + '\\' + fileName, 0)
        i += 1
        
    # Add spacing information in nrrd header    
    spacings = [patientDict['Pixel Spacing x'], patientDict['Pixel Spacing y'], patientDict['Spacing Between Slices']]
    header = {'spacings' : spacings}

    nrrd.write(dataPath + '.nrrd', data, header) # Write 3D-nrrd-file
    return 1

def create_masks_and_nrrds(dataPath, overWrite = False, readGray = True):
    """ 
    ACTION: 
        Creates a mask for every tiff image in folder and subfolders (not for images with extension "_mask"). 
        Creates nrrd files for every tiff image and its mask.
        The masks are named with the same name as the original images but with an extension "_mask"
    INPUTS: 
        dataPath
        overwrite: overwrite existing masks and nrrd files
        readGray: create nrrd files based on gray scale image
    """

    # Read file names of manually created masks from manual_masks.txt
    manualMasksPath = dataPath + '\\manual_masks.txt'
    with open(manualMasksPath, 'r') as manualMasksFile:
        manualMaskNames = manualMasksFile.read().splitlines()

    # Read Dicom files
    allPatientsDict = extract_dicom_features(dataPath)

    patDirNames = os.listdir(dataPath)
    for patDirName in patDirNames:

        createdMasks = False
        createdNrrds = False

        patDirPath = dataPath + '\\' + patDirName
        if os.path.isdir(patDirPath):

            # This line will be reached once for every patient directory
            patSubDirNames = os.listdir(patDirPath)
            for patSubDirName in patSubDirNames:

                patSubDirPath = patDirPath + '\\' + patSubDirName
                if os.path.isdir(patSubDirPath) and re.search('^Pat.+T2M[+frisk]*$', patSubDirName): 
                    # This line will be reached once for sub-directory starting with 'Pat' 
                    # and ending with 'T2M', 'T2M+' or 'T2Mfrisk' (e.i. all folders with green line segmentations)

                    # Create directory for masks if it does not exsist
                    if not os.path.isdir(patSubDirPath + '_mask'):
                        os.mkdir(patSubDirPath + '_mask')

                    fileNameExts = os.listdir(patSubDirPath)
                    for fileNameExt in fileNameExts:
                        fileName, fileExt = os.path.splitext(fileNameExt)
                        if fileExt == '.tiff':
                            # This line will be reached for all tiff-files to be masked
                            imagePath = patSubDirPath + '\\' + fileNameExt
                            maskPath = patSubDirPath + '_mask\\' + fileName + '_mask' + fileExt
                            if not os.path.exists(maskPath) or (overWrite and fileName + '_mask' + fileExt not in manualMaskNames):
                                createdMasks = create_mask(imagePath, maskPath) or createdMasks

                    if not os.path.exists(patSubDirPath + '_mask.nrrd') or overWrite:
                        try:
                            patientDict = allPatientsDict[patDirName[3:]]
                            createdNrrds = create_3d_nrrd(patSubDirPath + '_mask', patientDict) or createdNrrds
                        except:
                            print(patDirName, ': No nrrd-file created (DICOM missing)')

                elif os.path.isdir(patSubDirPath) and re.search('^Pat.+T2U$', patSubDirName): # THIS IS CHANGED TO ONLY CREATE NRRDS FOR T2-images!!!!!!!!
                    # This line will be reached once for sub-directory starting with 'Pat' and ending with 'T2U'

                    if not os.path.exists(patSubDirPath + '.nrrd') or overWrite:
                        try:
                            patientDict = allPatientsDict[patDirName[3:]]
                            createdNrrds = create_3d_nrrd(patSubDirPath, patientDict) or createdNrrds
                        except:
                            print(patDirName, ': No nrrd-file created (DICOM missing)')

        if createdNrrds and createdMasks:
            print(patDirName + ': Masks and nrrd-files created. ')
        elif createdMasks:
            print(patDirName + ': Masks created. ')
        elif createdNrrds:
            print(patDirName + ': Nrrd-files created. ')



def erosion_manual_masks(dataPath):
    """ 
    ACTION: 
        Perform erosion on all images listed in 'manual_masks_to_add.txt', overwrites the images with eroded ones.     
    INPUTS: 
        dataPath
    """

    # Read file names of manually created masks from manual_masks_to_add.txt
    manualMasksPath = dataPath + '\\manual_masks_to_add.txt'
    with open(manualMasksPath, 'r') as manualMasksFile:
        manualMaskNames = manualMasksFile.read().splitlines()

    patDirNames = os.listdir(dataPath)
    for patDirName in patDirNames:

        patDirPath = dataPath + '\\' + patDirName
        if os.path.isdir(patDirPath):

            # This line will be reached once for every patient directory
            patSubDirNames = os.listdir(patDirPath)
            for patSubDirName in patSubDirNames:

                patSubDirPath = patDirPath + '\\' + patSubDirName
                if os.path.isdir(patSubDirPath) and re.search('^Pat.+T2M[+frisk]*_mask$', patSubDirName): 
                    # This line will be reached once for sub-directory starting with 'Pat',
                    # containing 'T2M', 'T2M+' or 'T2Mfrisk' and ending with '_mask'

                    fileNameExts = os.listdir(patSubDirPath)

                    for fileNameExt in fileNameExts:
                        if fileNameExt in manualMaskNames:
                            # This line will be reached for files listed in manual_masks_to_add.txt
                            maskPath = patSubDirPath + '\\' + fileNameExt
                            mask = cv2.imread(maskPath) # Read
                            mask = erosion(mask) # Perform erosion
                            cv2.imwrite(maskPath, mask) # Write

def create_manual_masks(dataPath):
    """ 
    ACTION: 
        Create the masks listed in 'manual_masks_to_add.txt'. The masks are without interior points and erosion.   
    INPUTS: 
        dataPath
    """

    # Read file names of manually created masks from manual_masks_to_add.txt
    manualMasksPath = dataPath + '\\manual_masks_to_add.txt'
    with open(manualMasksPath, 'r') as manualMasksFile:
        manualMaskNames = manualMasksFile.read().splitlines()

    # Remove '_mask' from manualMaskNames
    manualMaskNames = [x.replace('_mask', '') for x in manualMaskNames if len(x)>3]

    nrFoundFiles = 0 # Count the number of files found matching files in manualMaskNames

    patDirNames = os.listdir(dataPath)
    for patDirName in patDirNames:

        patDirPath = dataPath + '\\' + patDirName
        if os.path.isdir(patDirPath):

            # This line will be reached once for every patient directory
            patSubDirNames = os.listdir(patDirPath)
            for patSubDirName in patSubDirNames:

                patSubDirPath = patDirPath + '\\' + patSubDirName
                if os.path.isdir(patSubDirPath) and re.search('^Pat.+T2M[+frisk]*$', patSubDirName): 
                    # This line will be reached once for sub-directory starting with 'Pat',
                    # containing 'T2M', 'T2M+' or 'T2Mfrisk' and ending with '_mask'

                    fileNameExts = os.listdir(patSubDirPath)

                    for fileNameExt in fileNameExts:
                        if fileNameExt in manualMaskNames:
                            # This line will be reached for files listed in manual_masks_to_add.txt
                            fileName, fileExt = os.path.splitext(fileNameExt)
                            imagePath = patSubDirPath + '\\' + fileNameExt
                            maskPath = patSubDirPath + '_mask\\' + fileName + '_mask' + fileExt
                            create_mask(imagePath, maskPath, addInterior=False) # Creata mask without interior
                            nrFoundFiles += 1

    if not nrFoundFiles == len(manualMaskNames):
        print('WARNING: Number of files found with names from \'manual_masks_to_add.txt\' does not match the number of files listed in \'manual_masks_to_add.txt\'.')
        print('Number of files found: ', nrFoundFiles)
        print('Number of listed files:', len(manualMaskNames))



def extract_dicom_feature(dicomPath, feature):
    """ 
    ACTION: 
        Reads DICOM file, searches for the feature and outputs the value 
    INPUTS: 
        dicomPath: path to dicom file
        feature: the feature of interest, note that it is case sensitive
    OUTPUT:
        value of the given feature, 0 if feature can't be found in the file
    """

    htmlData = open(dicomPath, 'rb').read().decode('utf-16') #Open file in correct format

    # Extract string contain feature, everything inbetween, and the value
    tdfeatureString = re.findall('<td>' + feature + '</td><td>.*?</td><td>.*?</td><td>.*?</td><td>.*?</td><td>.*?</td>', htmlData)

    # Check that the feature was found
    if len(tdfeatureString) > 0:
        tdValue = re.findall('<td>.*?</td>', tdfeatureString[0])  # List all <td> </td> found in the string, the last value is the one of interest
        value = re.sub('</*td>', '', tdValue[-1]) # remove <td> and </td> and get the last value
        return value  

    # Print text and return 0 if the extraction failed.
    else:
        print(f'Could not find {feature} in HTML file')
        return 0

def extract_dicom_features(dataPath, features = ['Patients Sex', 'Patients Weight', 'Pixel Spacing', 'Spacing Between Slices']):
    """
    ACTION: 
        Reads DICOM files for all patients, creates a csv with content of interest and returns a dictionary for all patients with the same content.
    INPUTS: 
        dataPath
        features: list of features to extract from DICOM-files
    OUTPUT:
        A dictionary containing dictionaries for each patient with its DICOM features 
    """
    # Create a dictionary which will contain a dictionary for each patient
    allPatsDict = {}

    # Create file for output
    outcomePath = "../../patient_data/dicom_features.csv"
    with open(outcomePath, "w") as f:
        f.truncate() 

    # Create list with all existing patient IDs in the data folder
    folderContent = os.listdir(dataPath)
    patIds = [x for x in folderContent if re.search('^Pat[0-9]?[0-9]?[0-9]$', x)]


    # Go through all the patients
    for patientId in patIds:
        dicomPath = "../../patient_data/DICOM/" + patientId + "T2.HTML" #path to html file
        patDict = {} # new dictionary

        # Check that we have HTML file
        if os.path.isfile(dicomPath):
            for feature in features:
                patDict[feature] = extract_dicom_feature(dicomPath, feature) # Read HTML file

            # Some readings requires manual adjustments
            if 'Pixel Spacing' in features:
                pixelSpacing = patDict['Pixel Spacing'].split('\\')
                patDict['Pixel Spacing x'] = pixelSpacing[0]
                patDict['Pixel Spacing y'] = pixelSpacing[1]
                
            if 'Patients Sex' in features:
                patSex = patDict['Patients Sex']
                patDict['Patients Sex'] = 0 if patSex == 'M' else 1

            # Store patient dict in all patients dict
            allPatsDict[patientId[3:]] = patDict

        # Header in CSV is the keys from the dictionary
        header = list(patDict.keys()) 

        # Check if file is empty, if it is we also create the header
        if os.path.getsize(outcomePath) == 0:
            with open(outcomePath, 'w', newline='') as File:
                writer = csv.DictWriter(File, fieldnames=header, delimiter = ';')
                writer.writeheader()
                writer.writerow(patDict)

        # If file is not empty we append the new content
        else:
            with open(outcomePath, 'a', newline='') as File:
                writer = csv.DictWriter(File, fieldnames=header, delimiter = ';')
                writer.writerow(patDict)

    return allPatsDict
