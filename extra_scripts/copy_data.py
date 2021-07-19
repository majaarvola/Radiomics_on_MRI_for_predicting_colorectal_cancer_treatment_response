import os
import shutil

dst = "D:\colorectal_cancer_data_from_emil"
src = "..\..\..\patient_data"

patfolders = [f for f in os.listdir(src) if "Pat" in f]

for f in sorted(patfolders):
    os.makedirs(os.path.join(dst, f), exist_ok=True)
    for sub_folder in os.listdir(os.path.join(src, f)):
        if sub_folder.endswith("_mask"):
            print(f"Copying '{sub_folder}'")
            shutil.copytree(os.path.join(src, f, sub_folder), os.path.join(dst, f, sub_folder))
        elif os.path.isdir(os.path.join(src, f, sub_folder)):
            print(f"Creating empty folder '{sub_folder}'")
            os.makedirs(os.path.join(dst, f, sub_folder), exist_ok=True)
