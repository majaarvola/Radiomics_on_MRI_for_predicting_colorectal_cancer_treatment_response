
import numpy as np
import nrrd
import radiomics
import cv2
import os


def create_mask(inputFile, outputFile, showResult=True):
    """ 
    Generates a black and white mask of the input image
    The white area corresponds to green markings in the
    file including any interior points and the rest is black.
    """

    # Read image (if it exists) and make copy for comparison
    if cv2.haveImageReader(inputFile):
        original_image = cv2.imread(inputFile)
    else:
        print(f"Failed to read input file at {inputFile}")
        return
    image = original_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range and mask everything within range
    lower = np.array([5, 10, 5], dtype="uint8")
    upper = np.array([230, 255, 230], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    # Add interior points to mask
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = cv2.fillPoly(mask, cnts, (255, 255, 255))

    # save the output
    if not cv2.imwrite(outputFile, mask):
        print(f"Failed to write output file at {outputFile}")

    if showResult:
        # Apply mask to original image and show images for visual verification
        result = cv2.bitwise_and(original_image, original_image, mask=mask)
        cv2.imshow('intial', original_image)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        cv2.waitKey(0)

def img2nrrd(imageFile):
    """
    ACTION: Creates a file with the extension '.nrrd'
    INPUTS: imageFile (if no extension, '.tiff' is tried)
    """

    filename, file_extension = os.path.splitext(imageFile)
    outputFile = filename + ".nrrd"
    if len(file_extension) == 0:
        imageFile += ".tiff"

    if not os.path.exists(outputFile):
        if cv2.haveImageReader(imageFile):
            nrrd.write(outputFile, cv2.imread(imageFile))
    else:
        print("File already exists: ", outputFile)

def create_all_nrrd(folderPath):
    """
    ACTION: Creates nrrd-files for every image in folder and subfolders
    """
    pass

def create_masks(folderPath):
    """ 
    ACTION: Creates a mask for every image in folder and subfolders (not for images with extension "_mask"). 
            The masks are named with the same name as the original images but with an extension "_mask"
    INTPUT: folderpath
    """
    pass
