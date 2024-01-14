import os
import shutil

root_name = "dataset"
image_folder_paths = [f"{root_name}/Train/Female", f"{root_name}/Train/Male",
                      f"{root_name}/Validation/Female", f"{root_name}/Validation/Male",
                      f"{root_name}/Test/Female", f"{root_name}/Test/Male"]

MERGED_FOLDERS_PATH = "merged_dataset"
MALE_PATH = os.path.join(MERGED_FOLDERS_PATH, "male")
FEMALE_PATH = os.path.join(MERGED_FOLDERS_PATH, "female")
os.makedirs(MALE_PATH, exist_ok=True)
os.makedirs(FEMALE_PATH, exist_ok=True)

male_name_counter, female_name_counter = 0, 0
for path in image_folder_paths:
    img_names = os.listdir(path)
    for img_name in img_names:
        img_path = os.path.join(path, img_name)

        

        # rename the image
        if "Female" in path:
            new_img_name = str(female_name_counter).zfill(6) + '.png'
            new_img_path = os.path.join(FEMALE_PATH, new_img_name)
            female_name_counter += 1
        elif "Male" in path:
            new_img_name = str(male_name_counter).zfill(6) + '.png'
            new_img_path = os.path.join(MALE_PATH, new_img_name)
            male_name_counter += 1
        
        shutil.copy(img_path, new_img_path)
    
    print(f"Finished {path}!")