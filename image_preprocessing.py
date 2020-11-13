
import numpy as np
import nrrd
import radiomics
import cv2
import os
import re

def create_mask(imageFile, maskFile, showResult=False):
    """ 
    ACTION: Generates a black and white mask of the input image
            The white area corresponds to green markings in the
            file including any interior points and the rest is black.
    INPUTS: imageFile: path to image file
            maskFile: path of mask file to be created
            showResult: display image, mask, segmented image
    OUTPUT: 1 if mask was created, 0 if not
    """

    # Read image (if it exists) and make copy for comparison
    if cv2.haveImageReader(imageFile):
        originalImage = cv2.imread(imageFile)
    else:
        print(f"Failed to read input file at {imageFile}")
        return 0
    image = originalImage.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range and mask everything within range
    lower = np.array([50, 125, 125], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    # Add interior points to mask
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = cv2.fillPoly(mask, cnts, (255, 255, 255))

    # Perform erosion on mask
    mask = erosion(mask)

    # save the output
    if not cv2.imwrite(maskFile, mask):
        print(f"Failed to write output file at {maskFile}")

    if showResult:
        # Apply mask to original image and show images for visual verification
        result = cv2.bitwise_and(originalImage, originalImage, mask=mask)
        cv2.imshow('intial', originalImage)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        cv2.waitKey(0)
    
    return 1

def erosion(mask):
    """
    ACTION: Performs erosion on the input image, returns the result
    INPUTS: mask: image to perform erosion on
    """
    kernel = np.ones((3,3),np.uint8)
    kernel[0,0] = 0
    kernel[0,-1] = 0
    kernel[-1,0] = 0
    kernel[-1,-1] = 0
    return cv2.erode(mask,kernel,iterations = 1)


def create_3d_nrrd(folderPath):
    """
    ACTION: Create three dimensional nrrd-file from all tiff-files in given folder, 
            the nrrd-file will be located next to the folder with the same name
    INPUTS: folderPath: folder with tiff-files
    OUTPUT: 1 if nrrd-file was created, 0 if not
    """

    if not os.path.isdir(folderPath): return 0 # Exit if folder does not exist

    fileNames = os.listdir(folderPath) # Read content in the folder
    tiffFileNames = [x for x in fileNames if x.endswith('.tiff')] # List with all tiff-images
    nrTiffFiles = len(tiffFileNames) # Number of tiff-images in folder

    if nrTiffFiles == 0: return 0 # Exit if folder contains no tiff-files

    firstImage = cv2.imread(folderPath + '\\' + tiffFileNames[0], 0)
    w, h = firstImage.shape # Width and height of the first tiff-image in the folder

    # Make 3D-nrrd-file from all tiff-files in the folder
    i = 0
    data = np.zeros((w,h,nrTiffFiles)) # Allocate 3D-array for data, 400x400 IMAGES ASSUMED
    for fileName in sorted(tiffFileNames):
        # Insert grayscaled layer into 3D-array
        data[:,:,i] = cv2.imread(folderPath + '\\' + fileName, 0)
        i += 1

    nrrd.write(folderPath + '.nrrd', data) # Write 3D-nrrd-file
    return 1

def create_masks_and_nrrds(folderPath, overWrite = False, readGray = True):
    """ 
    ACTION: Creates a mask for every tiff image in folder and subfolders (not for images with extension "_mask"). 
            Creates nrrd files for every tiff image and its mask.
            The masks are named with the same name as the original images but with an extension "_mask"
            
    INPUTS: folderpath
            overwrite: overwrite existing masks and nrrd files
            readGray: create nrrd files based on gray scale image
    """

    # Read file names of manually created masks from manual_masks.txt
    manualMasksPath = folderPath + '\\manual_masks.txt'
    with open(manualMasksPath, 'r') as manualMasksFile:
        manualMaskNames = manualMasksFile.read().splitlines()

    patDirNames = os.listdir(folderPath)
    for patDirName in patDirNames:

        createdMasks = False
        createdNrrds = False

        patDirPath = folderPath + '\\' + patDirName
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
                                createdMasks = create_mask(imagePath, maskPath, showResult=False) or createdMasks

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



def erosion_manual_masks(folderPath):
    """ 
    ACTION: Perform erosion on all images listed in manual_masks.txt, overwrites the images with eroded ones. 
            
    INPUTS: folderpath
    """

    # Read file names of manually created masks from manual_masks.txt
    manualMasksPath = folderPath + '\\manual_masks.txt'
    with open(manualMasksPath, 'r') as manualMasksFile:
        manualMaskNames = manualMasksFile.read().splitlines()

    patDirNames = os.listdir(folderPath)
    for patDirName in patDirNames:

        patDirPath = folderPath + '\\' + patDirName
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
