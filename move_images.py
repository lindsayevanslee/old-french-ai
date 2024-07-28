# Description: This script moves all the images in a folder to a new folder with the same name as the image.

import os
import shutil

# Set the folder path where all output will be stored
folder_path = 'data/ME MSS Images/png'

def put_files_in_own_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Create a new folder with the same name as the file
            folder_name = os.path.splitext(file)[0]
            new_folder_path = os.path.join(root, folder_name)
            print(f"New folder: {new_folder_path}")
            os.makedirs(new_folder_path, exist_ok=True)

            # Move the file into the new folder
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_folder_path, "0_converted_to_png.png")
            print(f"Moving {old_file_path} to {new_file_path}")
            shutil.move(old_file_path, new_file_path)

put_files_in_own_folder(folder_path)


def move_files_to_new_folders(old_folder, new_folder, new_file_name):
    for root, dirs, files in os.walk(old_folder):
        for file in files:
            print(f"Root: {root}")
            print(f"File: {file}")

            #create new folder by replacing old_folder with new_folder in root
            new_folder_path = os.path.join(root.replace(old_folder, new_folder), os.path.splitext(file)[0])
            print(f"New folder: {new_folder_path}")

            # Move the file into the new folder
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_folder_path, new_file_name)
            print(f"Moving {old_file_path} to {new_file_path}")
            shutil.move(old_file_path, new_file_path)


move_files_to_new_folders('data/ME MSS Images/png_resized', folder_path, "01_resized.png")
move_files_to_new_folders('data/ME MSS Images/binary', folder_path, "02_binary.png")
move_files_to_new_folders('data/ME MSS Images/skew_corrected', folder_path, "03_skew_corrected.png")





