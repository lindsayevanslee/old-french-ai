import os
import shutil


#set the input directory
input_dir = 'data/ME MSS Images/output' 

#set the file names to move
file_names = ['02_binary.png', '03_skew_corrected.png']

#function to create "old" folder and move files to that folder
def move_files_to_old_folder(directory, file_names):
    for root, dirs, files in os.walk(directory):
        if files:
            # Create the "old" folder within the innermost directory
            old_folder = os.path.join(root, "old")
            os.makedirs(old_folder, exist_ok=True)

            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    shutil.move(file_path, old_folder)
                    print(f"Moved {file_path} to {old_folder}")

#run function
move_files_to_old_folder(input_dir, file_names)