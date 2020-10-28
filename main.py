import numpy as np
import cv2


def create_mask(inputFile, outputFile, showResult=True):
    """ Generates a black and white mask of the input image
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
        cv2.waitKey(1000)


if __name__ == "__main__":
    #example run:
    inputFile = "Pat12T2M45.tiff"
    outputFile = "Pat12T2M45_masked.tiff"
    create_mask(inputFile, outputFile, showResult=False)
