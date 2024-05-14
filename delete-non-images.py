import os

#Delete files with specific extensions
def delete_files_with_extensions(directory, extensions):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Set directory and extensions
directory = "data/ME MSS Images/heic"
extensions = ["DS_Stores", ".MOV", ".mov", ".docx"]  # Add your desired file extensions here
delete_files_with_extensions(directory, extensions)