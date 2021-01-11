# Radiomics on MRI for predicting colorectal cancer treatment response

Inledande info

## Walk-through how to implement pipeline in an example (typ main)

## How to structure data in your folder
The code is highly dependent on that the data follows a specific structure. Several functions takes the input argument `dataPath`, which is the path to the folder containg all the data. This data folder must contain data in the following format:

* **Patient folders**: there should be one folder per patient named `PatX` where X is the patient ID. Each patient folder contains following subfolders: 
  * Folder with original MRI, seperate tiff-images for each layer. Sorted alphabetically, the file order should correspond to the physical ordering. Foldername must start with `Pat` and end with `T2U`.
  * Folder with MRI with tumor segmentation in green pixels, seperate tiff-images for each layer. Sorted alphabetically, the file order should correspond to the physical ordering. Foldername must start with `Pat` and end with `T2M`, `T2M+` or `T2Mfrisk`. It is possible to have multiple folders.
* **DICOM folder**: folder named `DICOM` containing HTML files with DICOM header data for each patient. These files should be named `PatXT2` where X is the patient ID.
* **Manual features file**: a csv file with the features that were needed to add manually, one row per patient. If any information is missing for a patient, that patient will be excluded. The file should contain the following headers:
  * id: patient id.
  * outcome: response to treatment, stated as positive integers. Missing information is noted as -1.
  * age: age of patient. Missing information is noted as -1.
  * treatment: 0 or 1 depending on which treatment a patient has received. Missing information is noted as -1.


## Image processing: Correcting failing masks
When binary masks are automatically created from segmented images, there is a risk that the algorithm is unable to generate the correct mask. It is therefor necessary to compare the segmented images with corresponding masks and correct the failing ones. In this section, an approach for correcting failing masks is suggested. 
1.	Create masks automatically, if not done already, by executing the function `create_masks_and_nrrds`. 
2.	Compare all masks with corresponding segmented image. When you encounter a failing mask, add its filename (with extension) to a new row in a file that you name `manual_masks_to_add.txt` and place directly in your data folder. 
3.	Execute `create_manual_masks`, and all masks listed in `manual_masks_to_add.txt` will be created without adding the interior, and without performing erosion. I.e., only the line itself will be labeled with white pixels. 
4.	Open each mask created in step 3 in an image editor, fill the interior with white pixels, and save the image. 
5.	Execute `erosion_manual_masks`, and all masks listed in `manual_masks_to_add.txt` will undergo erosion. 
6.	Rename the file `manual_masks_to_add.txt` to `manual_masks.txt`. The masks listed in `manual_masks.txt` will not be overwritten when executing `create_masks_and_nrrds` with `overwrite=True`. 
