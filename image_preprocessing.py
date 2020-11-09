
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
    """

    # Read image (if it exists) and make copy for comparison
    if cv2.haveImageReader(imageFile):
        originalImage = cv2.imread(imageFile)
    else:
        print(f"Failed to read input file at {imageFile}")
        return
    image = originalImage.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range and mask everything within range
    lower = np.array([50, 125, 125], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    # Do not create mask if there is no segmentation in the image
    if np.max(mask) == 0:
        return 0

    # Add interior points to mask
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = cv2.fillPoly(mask, cnts, (255, 255, 255))

    # Perform erosion on mask
    kernel = np.ones((3,3),np.uint8)
    kernel[0,0] = 0
    kernel[0,-1] = 0
    kernel[-1,0] = 0
    kernel[-1,-1] = 0
    mask = cv2.erode(mask,kernel,iterations = 1)

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

def create_nrrd(imageFile, readGray = True):
    """
    ACTION: Creates a file with the extension '.nrrd'
    INPUTS: imageFile: path to image file, if no extension, '.tiff' is tried
            readGray: create nrrd file based on gray scale image
    """

    filename, file_extension = os.path.splitext(imageFile)
    nrrdFile = filename + ".nrrd"
    if len(file_extension) == 0:
        imageFile += ".tiff"

    if cv2.haveImageReader(imageFile):
        nrrd.write(nrrdFile, cv2.imread(imageFile, 0 if readGray else 1))
        return 1

    return 0



def create_masks_and_nrrds(folderPath, overWrite = False, readGray = True):
    """ 
    ACTION: Creates a mask for every tiff image in folder and subfolders (not for images with extension "_mask"). 
            Creates nrrd files for every tiff image and its mask.
            The masks are named with the same name as the original images but with an extension "_mask"
            
    INPUTS: folderpath
            overwrite: overwrite existing masks and nrrd files
            readGray: create nrrd files based on gray scale image
    """

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
                            if not os.path.exists(maskPath) or overWrite:
                                createdMasks = createdMasks or create_mask(imagePath, maskPath, showResult=False)
                            if not os.path.exists(patSubDirPath + '_mask\\' + fileName + '_mask.nrrd') or overWrite:
                                createdNrrds = createdNrrds or create_nrrd(maskPath, readGray)

                elif os.path.isdir(patSubDirPath) and re.search('^Pat.+U$', patSubDirName):
                    # This line will be reached once for sub-directory starting with 'Pat' and ending with 'U'

                    fileNameExts = os.listdir(patSubDirPath)
                    for fileNameExt in fileNameExts:
                        fileName, fileExt = os.path.splitext(fileNameExt)
                        if fileExt == '.tiff':
                            # This line will be reached for all tiff-files to generate nrrd-file from
                            imageFile = patSubDirPath + '\\' + fileName
                            if not os.path.exists(imageFile + '.nrrd') or overWrite:
                                createdNrrds = createdNrrds or create_nrrd(imageFile + fileExt, readGray)

        if createdNrrds and createdMasks:
            print(patDirName + ': Masks and nrrd-files created. ')
        elif createdMasks:
            print(patDirName + ': Masks created. ')
        elif createdNrrds:
            print(patDirName + ': Masks nrrd-files created. ')
