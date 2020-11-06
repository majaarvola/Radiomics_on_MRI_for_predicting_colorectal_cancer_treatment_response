
import numpy as np
import nrrd
import radiomics
import cv2
import os


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


def create_masks_and_nrrds(folderPath, overWrite = False, readGray = True):
    """ 
    ACTION: Creates a mask for every tiff image in folder and subfolders (not for images with extension "_mask"). 
            Creates nrrd files for every tiff image and its mask.
            The masks are named with the same name as the original images but with an extension "_mask"
            
    INPUTS: folderpath
            overwrite: overwrite existing masks and nrrd files
            readGray: create nrrd files based on gray scale image
    """
    for dirPath, _, files in os.walk(folderPath):
        createdMasks = False
        createdNrrds = False
        for fileNameExt in files:
            fileName, fileExt = os.path.splitext(fileNameExt)
            if fileExt == ".tiff" and not fileName.endswith("_mask"):
                maskFile = dirPath + "\\" + fileName + "_mask" 
                imageFile = dirPath + "\\" + fileName

                # Create masks
                if not os.path.exists(maskFile + fileExt) or overWrite:
                    create_mask(imageFile + fileExt, maskFile + fileExt, showResult=False)
                    createdMasks = True

                # Create nrrds
                if not os.path.exists(imageFile + ".nrrd") or overWrite:
                    create_nrrd(imageFile, readGray)
                    createdNrrds = True

                if not os.path.exists(maskFile + ".nrrd") or overWrite:
                    create_nrrd(maskFile, readGray)
                    createdNrrds = True
                
        if createdMasks and createdNrrds: print(f'Created masks and nrrd files in "{dirPath}"')
        elif createdMasks: print(f'Created masks in "{dirPath}"')
        elif createdNrrds: print(f'Created nrrd files in "{dirPath}"')
