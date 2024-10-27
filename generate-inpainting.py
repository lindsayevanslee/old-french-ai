import torch
from inpainting_model import UNetInpaint
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def load_and_process_image(image_path, mask_path, img_size=512):
    # Load images
    mutilated = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    mutilated = transform(mutilated)
    mask = transform(mask)
    
    # Create mask (same as training)
    mask_gray = mask.mean(dim=0)
    mask_binary = (mask_gray > 0).float().unsqueeze(0)
    
    # Combine image and mask
    input_tensor = torch.cat([mutilated, mask_binary], dim=0)
    return input_tensor.unsqueeze(0)  # Add batch dimension

def generate_inpainting():
    # Load the trained model
    model = UNetInpaint()
    model.load_state_dict(torch.load('models/unet_inpaint.pth'))
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load and process your test image
    input_tensor = load_and_process_image(
        'path/to/mutilated_image.png',
        'path/to/mask_image.png'
    ).to(device)
    
    # Generate inpainting
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output tensor to image
    output_image = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Plot or save
    plt.imshow(output_image)
    plt.axis('off')
    plt.savefig('inpainted_result.png')
    plt.show()

if __name__ == "__main__":
    generate_inpainting()