
import numpy as np
import nrrd
import radiomics
import cv2
import os
import re

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


def create_3d_nrrd(dataPath):
    """
    ACTION: 
        Create three dimensional nrrd-file from all tiff-files in given folder, 
        the nrrd-file will be located next to the folder with the same name
    INPUTS: 
        dataPath: folder with tiff-files
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

    nrrd.write(dataPath + '.nrrd', data) # Write 3D-nrrd-file
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
                        createdNrrds = create_3d_nrrd(patSubDirPath + '_mask') or createdNrrds

                elif os.path.isdir(patSubDirPath) and re.search('^Pat.+U$', patSubDirName):
                    # This line will be reached once for sub-directory starting with 'Pat' and ending with 'U'

                    if not os.path.exists(patSubDirPath + '.nrrd') or overWrite:
                        createdNrrds = create_3d_nrrd(patSubDirPath) or createdNrrds

        if createdNrrds and createdMasks:
            print(patDirName + ': Masks and nrrd-files created. ')
        elif createdMasks:
            print(patDirName + ': Masks created. ')
        elif createdNrrds:
            print(patDirName + ': Masks nrrd-files created. ')



def erosion_manual_masks(dataPath):
    """ 
    ACTION: 
        Perform erosion on all images listed in manual_masks.txt, overwrites the images with eroded ones.     
    INPUTS: 
        dataPath
    """

    # Read file names of manually created masks from manual_masks.txt
    manualMasksPath = dataPath + '\\manual_masks.txt'
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
                            # This line will be reached for files listed in manual_masks.txt
                            maskPath = patSubDirPath + '\\' + fileNameExt
                            mask = cv2.imread(maskPath) # Read
                            mask = erosion(mask) # Perform erosion
                            cv2.imwrite(maskPath, mask) # Write

def create_manual_masks(dataPath):
    """ 
    ACTION: 
        Perform erosion on all images listed in manual_masks.txt, overwrites the images with eroded ones.     
    INPUTS: 
        dataPath
    """

    # Read file names of manually created masks from manual_masks.txt
    manualMasksPath = dataPath + '\\manual_masks.txt'
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
                            # This line will be reached for files listed in manual_masks.txt
                            fileName, fileExt = os.path.splitext(fileNameExt)
                            imagePath = patSubDirPath + '\\' + fileNameExt
                            maskPath = patSubDirPath + '_mask\\' + fileName + '_mask' + fileExt
                            create_mask(imagePath, maskPath, addInterior=False) # Creata mask without interior
                            nrFoundFiles += 1

    if not nrFoundFiles == len(manualMaskNames):
        print('WARNING: Number of files found with names from \'manual_masks.txt\' does not match the number of files listed in \'manual_masks.txt\'.')
        print('Number of files found: ', nrFoundFiles)
        print('Number of listed files:', len(manualMaskNames))
