import os
from PIL import Image
from pillow_heif import register_heif_opener

input_dir = 'data/ME MSS Images/heic'
output_dir = 'data/ME MSS Images/png'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

#print all files present in the input directory
def print_files_recursively(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

print_files_recursively(input_dir)

#print all file extensions present in the input directory
def print_file_extensions(directory):
    extensions = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            extensions.add(os.path.splitext(file)[1])
    print(extensions)

print_file_extensions(input_dir)

# Register the HEIF file format with Pillow
register_heif_opener()

# Loop through all files recursively in the input directory
for root, dirs, files in os.walk(input_dir):
    print(f"root: {root}")
    print(f"dirs: {dirs}")
    #print(f"files: {files}")
    for filename in files:
        if filename.endswith('.HEIC') or filename.endswith('.heic'):
            # Create the output directory if it doesn't exist
            new_root = root.replace('heic', 'png')
            os.makedirs(new_root, exist_ok=True)
            
            # Get heic path
            heic_path = os.path.join(root, filename)
            #print(heic_path)

            # Get the PNG filename and path
            png_filename = os.path.splitext(filename)[0] + '.png'
            #print(png_filename)
            png_path = os.path.join(new_root, png_filename)
            #print(png_path)

            # If the PNG file already exists, skip the conversion
            if os.path.exists(png_path):
                print(f"File already exists: {png_path}")
                continue
            else:
                print(f"Converting: {heic_path} to {png_path}")

            # Open the HEIC file
            heic_image = Image.open(heic_path)

            # Convert the image to PNG format
            png_image = heic_image.convert('RGB')
            
            # Save the PNG image in the output directory
            png_image.save(png_path, 'PNG')

            # Close the image
            heic_image.close()


        elif filename.endswith('.jpeg') or filename.endswith('.jpg'):
            
            # Create the output directory if it doesn't exist
            new_root = root.replace('heic', 'png')
            os.makedirs(new_root, exist_ok=True)
            
            # Get the JPEG path
            jpeg_path = os.path.join(root, filename)
            #print(jpeg_path)

            # Get the PNG filename and path
            png_filename = os.path.splitext(filename)[0] + '.png'
            #print(png_filename)
            png_path = os.path.join(new_root, png_filename)
            #print(png_path)

            # If the PNG file already exists, skip the conversion
            if os.path.exists(png_path):
                print(f"File already exists: {png_path}")
                continue
            else:
                print(f"Converting: {jpeg_path} to {png_path}")

            # Open the JPEG file
            jpeg_image = Image.open(jpeg_path)

            # Save the JPEG image in the output directory
            jpeg_image.save(png_path, 'PNG')

            # Close the image
            jpeg_image.close()

        elif filename.endswith('.png'):

            # Create the output directory if it doesn't exist
            new_root = root.replace('heic', 'png')
            os.makedirs(new_root, exist_ok=True)

            #Copy the PNG file to the output directory
            png_path_orig = os.path.join(root, filename)
            #print(png_path_orig)

            # Get the output PNG filename and path
            png_filename = os.path.splitext(filename)[0] + '.png'
            #print(png_filename)
            png_path = os.path.join(root.replace('heic', 'png'), png_filename)
            #print(png_path)


            # If the PNG file already exists, skip the conversion
            if os.path.exists(png_path):
                print(f"File already exists: {png_path}")
                continue
            else:
                print(f"Copying: {png_path_orig} to {png_path}")

            #Copy the PNG image in the output directory
            os.system(f'cp "{png_path_orig}" "{png_path}"')





print('Conversion complete!')

