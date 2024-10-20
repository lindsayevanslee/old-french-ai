from pdf2image import convert_from_path
import os

def pdf_to_jpeg(pdf_path, output_folder):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path)
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save each image as a JPEG file
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.jpeg')
        image.save(image_path, 'JPEG')
        print(f'Saved {image_path}')

if __name__ == "__main__":
    pdf_path = 'data/digitized versions/Vies des saints/original/Vies_de_saints_en_français__btv1b9063234c.pdf'
    output_folder = 'data/digitized versions/Vies des saints/jpeg'
    pdf_to_jpeg(pdf_path, output_folder)

    #delete page_1.jpeg and page_2.jpeg
    os.remove('data/digitized versions/Vies des saints/jpeg/page_1.jpeg')
    os.remove('data/digitized versions/Vies des saints/jpeg/page_2.jpeg')