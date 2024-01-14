import os
import shutil
import random

def split_balanced_data(source_folder, base_dest_folder, train_size=0.8, valid_size=0.1):
    categories = ['male', 'female']

    # Determine the minimum size among the categories
    min_size = min(len(os.listdir(os.path.join(source_folder, category))) for category in categories)

    # Calculate split sizes based on the minimum size
    train_count = int(min_size * train_size)
    valid_count = int(min_size * valid_size)
    test_count = min_size - train_count - valid_count

    for category in categories:
        # Create directories for train, valid, test splits
        os.makedirs(os.path.join(base_dest_folder, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(base_dest_folder, 'valid', category), exist_ok=True)
        os.makedirs(os.path.join(base_dest_folder, 'test', category), exist_ok=True)

        # List and shuffle images
        img_list = os.listdir(os.path.join(source_folder, category))
        random.shuffle(img_list)

        # Split and move/copy images
        for i, img in enumerate(img_list):
            if i >= min_size:  # Only use the first 'min_size' images
                break

            src_path = os.path.join(source_folder, category, img)
            if i < train_count:
                dest_path = os.path.join(base_dest_folder, 'train', category, img)
            elif i < train_count + valid_count:
                dest_path = os.path.join(base_dest_folder, 'valid', category, img)
            else:
                dest_path = os.path.join(base_dest_folder, 'test', category, img)

            # Copy image to the new directory
            shutil.copy(src_path, dest_path)

# Define source and destination folders
source_folder = 'gender_dataset_face'  # Update with your path
base_dest_folder = 'data2'  # Update with your desired path

# Run the function
split_balanced_data(source_folder, base_dest_folder)
