import os

# Print the files in the directories and file counts
def print_files_in_directories(folder_path):
    for root, dirs, files in os.walk(folder_path):
        print(f"Directory: {root}")
        print(f"Number of files: {len(files)}")
        print()

input_dir = 'data/ME MSS Images/test images folder structure/heic'
output_dir = 'data/ME MSS Images/test images folder structure/output'
print_files_in_directories(input_dir)
print_files_in_directories(output_dir)



#compare subdirectories across two root folders and see if they have the same file names and number of files (ignoring extensions)
def compare_folders(folder1, folder2):
    files1 = set()
    files2 = set()

    for root, dirs, files in os.walk(folder1):
        for file in files:
            file_name = os.path.splitext(file)[0]
            files1.add(file_name)

    for root, dirs, files in os.walk(folder2):
        for file in files:
            file_name = os.path.splitext(file)[0]
            files2.add(file_name)

    if len(files1) != len(files2):
        return False

    if files1 != files2:
        return False

    return True

def compare_subdirectories(folder1, folder2):
    for root, dirs, files in os.walk(folder1):
        subdirectory = root.replace(folder1, '').lstrip('/')
        result = compare_folders(os.path.join(folder1, subdirectory), os.path.join(folder2, subdirectory))
        print(f"Subdirectory: {subdirectory}")
        print(f"Result: {result}")
        print()


compare_subdirectories(input_dir, output_dir)






"""
Directory: data/ME MSS Images/heic
Number of files: 1

Directory: data/ME MSS Images/heic/L1-B
Number of files: 172

Directory: data/ME MSS Images/heic/O1-G
Number of files: 27

Directory: data/ME MSS Images/heic/T-A
Number of files: 3

Directory: data/ME MSS Images/heic/O-E
Number of files: 1

Directory: data/ME MSS Images/heic/O-E/Codex images
Number of files: 75

Directory: data/ME MSS Images/heic/O-E/Microfilm images
Number of files: 48

Directory: data/ME MSS Images/heic/O-C
Number of files: 79

Directory: data/ME MSS Images/heic/U-F
Number of files: 5

Directory: data/ME MSS Images/heic/Z-C
Number of files: 0

Directory: data/ME MSS Images/heic/T-F2
Number of files: 5

Directory: data/ME MSS Images/heic/Z-B
Number of files: 91

Directory: data/ME MSS Images/heic/T-C
Number of files: 97

Directory: data/ME MSS Images/heic/T-D
Number of files: 26

Directory: data/ME MSS Images/png
Number of files: 0

Directory: data/ME MSS Images/png/L1-B
Number of files: 172

Directory: data/ME MSS Images/png/O1-G
Number of files: 27

Directory: data/ME MSS Images/png/T-A
Number of files: 3

Directory: data/ME MSS Images/png/O-E
Number of files: 0

Directory: data/ME MSS Images/png/O-E/Codex images
Number of files: 75

Directory: data/ME MSS Images/png/O-E/Microfilm images
Number of files: 48

Directory: data/ME MSS Images/png/O-C
Number of files: 79

Directory: data/ME MSS Images/png/U-F
Number of files: 5

Directory: data/ME MSS Images/png/T-F2
Number of files: 3

Directory: data/ME MSS Images/png/Z-B
Number of files: 91

Directory: data/ME MSS Images/png/T-C
Number of files: 97

Directory: data/ME MSS Images/png/T-D
Number of files: 26
"""