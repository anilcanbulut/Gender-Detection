import os
import hashlib
import cv2
from tqdm import tqdm

def has_dublicate(current_hash, seen_hashes, file_path):
    if current_hash in seen_hashes:
        # Duplicate found, delete it
        os.remove(file_path)
        return True, seen_hashes
    else:
        seen_hashes[current_hash] = file_path
        return False, seen_hashes

def hash_image(image_path):
    """Compute the hash of an image using OpenCV."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray_image.tobytes()).hexdigest(), image

def find_duplicates(base_folder):
    """Find and delete duplicate images across different subfolders."""
    seen_hashes = {}
    subfolders = ['Train/Female', 'Train/Male', 'Validation/Female', 'Validation/Male', 'Test/Female', 'Test/Male']

    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        image_names = sorted(os.listdir(folder_path))
        num_of_imgs = len(image_names)
        deleted_count = 0
        print(f"{folder_path} contained {num_of_imgs} before.")
        for i in tqdm(range(num_of_imgs)):
            filename = image_names[i]
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, filename)
                image_hash, image = hash_image(file_path)

                ret, seen_hashes = has_dublicate(image_hash, seen_hashes, file_path)
                if ret:
                    deleted_count += 1
                
        print(f"{deleted_count} images were deleted from {subfolder}.\n")

# Use the function
base_folder = "dataset"  # Replace with the path to your dataset
find_duplicates(base_folder)
