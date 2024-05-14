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
        # Create the output directory if it doesn't exist
        new_root = root.replace('heic', 'png')
        os.makedirs(new_root, exist_ok=True)
        
        # Get path of original image
        orig_path = os.path.join(root, filename)
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
            print(f"Converting: {orig_path} to {png_path}")

        if filename.endswith('.png'):
            #Copy the PNG image in the output directory
            os.system(f'cp "{orig_path}" "{png_path}"')
            continue

        # Open the image
        orig_image = Image.open(orig_path)

        if filename.endswith('.HEIC') or filename.endswith('.heic'):
            
            # Convert the image to PNG format
            png_image = orig_image.convert('RGB')
            
            # Save the PNG image in the output directory
            png_image.save(png_path, 'PNG')
            
        elif filename.endswith('.jpeg') or filename.endswith('.jpg'):

            # Save the JPEG image in the output directory
            orig_image.save(png_path, 'PNG')
        
        # Close the image
        orig_image.close()


print('Conversion complete!')

