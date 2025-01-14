"""Apply inpainting from trained model"""

import torch
from inpainting_model import UNetInpaint
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def load_and_process_image(mutilated_path, excised_path, img_size=512):
    # Load images
    mutilated = Image.open(mutilated_path).convert('RGB')
    excised = Image.open(excised_path).convert('RGB')
    
    # Create mask from excised image
    mask = excised.point(lambda x: 255 if x > 0 else 0)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    mutilated = transform(mutilated)
    mask = transform(mask)
    
    # Create binary mask
    mask_gray = mask.mean(dim=0)
    mask_binary = (mask_gray > 0).float().unsqueeze(0)
    
    # Combine image and mask
    input_tensor = torch.cat([mutilated, mask_binary], dim=0)
    return input_tensor.unsqueeze(0)

def generate_inpainting(mutilated_path, excised_path):

    # Extract page number from the mutilated path
    page_number = mutilated_path.split('page_')[1].split('_')[0]


    # Load the trained model
    model = UNetInpaint()
    checkpoint = torch.load('models/unet_inpaint_epoch_40.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load and process images
    input_tensor = load_and_process_image(mutilated_path, excised_path).to(device)
    
    # Generate inpainting
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to image
    output_image = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original mutilated image
    axes[0].imshow(Image.open(mutilated_path))
    axes[0].set_title('Mutilated Image')
    axes[0].axis('off')
    
    # Mask (excised image)
    axes[1].imshow(Image.open(excised_path))
    axes[1].set_title('Excision Mask')
    axes[1].axis('off')
    
    # Inpainted result
    axes[2].imshow(output_image)
    axes[2].set_title('Inpainted Result')
    axes[2].axis('off')
    
    # Create directory if it doesn't exist
    save_dir = 'data/digitized versions/Vies des saints/model_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure
    plt.savefig(os.path.join(save_dir, f'page_{page_number}_inpainting_comparison.png'))
    plt.show()

if __name__ == "__main__":
    mutilated_path = 'data/digitized versions/Vies des saints/mutilations/page_14_mutilated.jpeg'
    excised_path = 'data/digitized versions/Vies des saints/excisions/page_14_excised.jpeg'
    generate_inpainting(mutilated_path, excised_path)