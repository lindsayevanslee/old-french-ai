import os
from PIL import Image
from tqdm import tqdm

#Set input directory
input_dir = 'data/ME MSS Images/output'

mb_limit = 10
bytes_to_mb = 1000 * 1000 #Finder uses base 10, but many programming languages use base 2

def reduce_image_size(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            #if file.endswith(".jpg") or file.endswith(".png"):
            if file == "0_converted_to_png.png":
                file_path = os.path.join(root, file)
                #file_size = os.path.getsize(file_path)
                file_size = os.stat(file_path).st_size


                #create path of output image
                #new_root = root.replace('png', 'png_resized')
                #os.makedirs(new_root, exist_ok=True)
                #output_path = os.path.join(new_root, file)
                output_path = os.path.join(root, "01_resized.png")
                print(output_path)
                

                print(f"File: {file}, Size: {round(file_size/bytes_to_mb, 1)} MB.")
                
                if file_size > mb_limit * bytes_to_mb:  
                    image = Image.open(file_path)
                    #image.thumbnail((1920, 1080))  # Adjust the size as needed
                    image.save(output_path, compress_level = 9)  # Adjust the quality as needed

                    #new_file_size = os.path.getsize(output_path)
                    new_file_size = os.stat(output_path).st_size
                    print(f"Reduced size of {file} from {round(file_size/bytes_to_mb, 1)} MB to {round(new_file_size/bytes_to_mb, 1)} MB.")
                else:
                    print(f"File {file} is already within the size limit: {round(file_size/bytes_to_mb, 1)} MB")

                    #Copy the PNG image in the output directory
                    os.system(f'cp "{file_path}" "{output_path}"')

                

# Call the function
reduce_image_size(input_dir)