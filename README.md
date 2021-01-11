# Radiomics on MRI for predicting colorectal cancer treatment response
The purpose of this project was to investigate if Radiomic features can be used to predict how patients with colorectal cancer will respond to preoperative radiation treatment. The effect of the treatment varies between patients and if the treatment respons for a patient could be predicted with high accuracy it would be helpful for medical planning. The project was conducted as a part of the course 'Project in Computational Science' at Uppsala University, with guidance from the Section of Radiology.

All the code in the project was written in Python version 3.6 and is publicly available in this repository.

## Create a pipeline
Here you will be guided on how to create your own pipeline. An example implementation can be found in `extra_scripts/example_main.py`.

* **Image processing**: To create binary masks and nrrd-files from your image data, execute `create_masks_and_nrrds` from `image_processing.py`. Note that manual adjustments of some masks need to be made, see separate section below. 
* **Extract features**: To extract Radiomic features and manual features such as age and outcome for each patient, execute `extract_features_from_all` from `feature_extraction.py`. Here you need to specify image type (only `T2` supported at the moment) and other settings for the extraction, see the documentation. 
* **Select feature subset**: Execute `select_features` from `feature_selection.py` to select a subset of features. Specify what method you want to use and its settings. Three different methods are supported: 
  *	MRMR
  *	LASSO
  *	Logistic Regression with L1-penalty
* **Train and evaluate prediction model**: Execute `create_evaluate_model` from `ml_prediction.py` to train a prediction model, evaluate it on training data, and test it on test data. The train/test-division of the data is hard coded in this function, you need to change this to fit your data. See the documentation to customize the training. If you want to save your results from evaluation and testing, you can use the function `write_results_to_csv`. Three methods for prediction are supported: 
    *	Random Forest regression
    *	Random Forest classification
    *	Logistic Regression



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
6.	Rename the file `manual_masks_to_add.txt` to `manual_masks.txt`. The masks listed in `manual_masks.txt` will not be overwritten when executing `create_masks_and_nrrds` with `overWrite=True`. 
7.	Execute `create_masks_and_nrrds` with `overWrite=True` to replace the incorrect nrrd-files created in step 1. 

## Contributors
* Emil Åberg ([emil96aberg](https://github.com/emil96aberg)) 
* Maja Arvola ([majaarvola](https://github.com/majaarvola))
