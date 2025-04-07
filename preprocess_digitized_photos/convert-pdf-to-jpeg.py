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
    pdf_path = 'data/digitized versions/Manuscrits numerises de la Bibliotheque municipale de Toulouse/original/Manuscrits_numérisés_de_la_Bibliothèque_pg1-500.pdf'
    output_folder = 'data/digitized versions/Manuscrits numerises de la Bibliotheque municipale de Toulouse/jpeg'
    pdf_to_jpeg(pdf_path, output_folder)
    
    # Delete specific pages if they exist
    for page in ['page_1.jpeg', 'page_2.jpeg']:
        page_path = os.path.join(output_folder, page)
        if os.path.exists(page_path):
            os.remove(page_path)