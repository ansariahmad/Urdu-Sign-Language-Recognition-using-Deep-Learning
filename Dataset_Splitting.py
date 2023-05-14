import os
import random
import shutil

# Define the paths to the train and test folders
train_path = "train/"
test_path = "test/"

# Create the test folder if it doesn't exist
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Loop through each folder in the train folder
for folder_name in os.listdir(train_path):
    # Define the paths to the source and destination folders
    src_folder = os.path.join(train_path, folder_name)
    dest_folder = os.path.join(test_path, folder_name)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get a list of all the image files in the source folder
    image_files = [f for f in os.listdir(src_folder) if f.endswith(".jpg")]

    # Randomly select 200 images to move to the test folder
    selected_files = random.sample(image_files, 200)

    # Move the selected files to the test folder
    for file_name in selected_files:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        shutil.move(src_file, dest_file)

    # Delete the moved files from the train folder
    for file_name in selected_files:
        file_path = os.path.join(src_folder, file_name)
        os.remove(file_path)
