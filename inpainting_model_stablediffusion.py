import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
from torchvision import transforms
import numpy as np

class ManuscriptDataset(Dataset):
    """Dataset class for medieval manuscript restoration"""
    def __init__(self, mutilated_dir, excised_dir, transform=None):
        """
        Args:
            mutilated_dir (str): Directory with mutilated images
            excised_dir (str): Directory with excised portions
            transform (callable, optional): Optional transform to be applied
        """
        self.mutilated_dir = mutilated_dir
        self.excised_dir = excised_dir
        self.transform = transform
        
        # Get all matching pairs of images
        self.image_pairs = []
        for filename in os.listdir(mutilated_dir):
            if filename.endswith('.jpeg'):
                excised_file = filename.replace('mutilated', 'excised')
                if os.path.exists(os.path.join(excised_dir, excised_file)):
                    self.image_pairs.append((filename, excised_file))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        mutilated_path = os.path.join(self.mutilated_dir, self.image_pairs[idx][0])
        excised_path = os.path.join(self.excised_dir, self.image_pairs[idx][1])
        
        # Load images
        mutilated_image = Image.open(mutilated_path).convert('RGB')
        excised_image = Image.open(excised_path).convert('RGB')
        
        # Create mask (white where content is missing)
        mask = create_mask(mutilated_image)
        
        if self.transform:
            mutilated_image = self.transform(mutilated_image)
            excised_image = self.transform(excised_image)
            mask = self.transform(mask)
        
        return mutilated_image, mask, excised_image

def create_mask(image):
    """Create binary mask indicating missing regions"""
    # Convert to grayscale and threshold to find very light regions
    gray = image.convert('L')
    threshold = 250  # Adjust based on your images
    mask = gray.point(lambda x: 255 if x > threshold else 0)
    return mask

def train_inpainting_model(model_path, dataset, num_epochs=10):
    """Fine-tune the stable diffusion model for manuscript restoration"""
    # Load pre-trained model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Training configuration
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)
    
    for epoch in range(num_epochs):
        for batch_idx, (mutilated, mask, target) in enumerate(dataset):
            # Move to GPU
            mutilated = mutilated.to("cuda")
            mask = mask.to("cuda")
            target = target.to("cuda")
            
            # Generate inpainted image
            with torch.no_grad():
                noise = torch.randn_like(mutilated)
                timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (mutilated.shape[0],))
                noisy_images = pipe.scheduler.add_noise(mutilated, noise, timesteps)
            
            # Forward pass
            noise_pred = pipe.unet(noisy_images, timesteps, encoder_hidden_states=None)["sample"]
            
            # Calculate loss
            loss = nn.MSELoss()(noise_pred, noise)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return pipe

def test_model(pipe, test_image_path, save_path="restored_output.png"):
    """Test the model on a single image and save the result"""
    # Load and preprocess test image
    test_image = Image.open(test_image_path).convert('RGB')
    mask = create_mask(test_image)
    
    # Prepare images for the model
    test_image = transforms.Resize((512, 512))(test_image)
    mask = transforms.Resize((512, 512))(mask)
    
    # Generate the inpainted image
    with torch.no_grad():
        output = pipe(
            image=test_image,
            mask_image=mask,
            guidance_scale=7.5,
            num_inference_steps=50
        ).images[0]
    
    # Save the result
    output.save(save_path)
    
    # Create a side-by-side comparison
    comparison = Image.new('RGB', (test_image.width * 3, test_image.height))
    comparison.paste(test_image, (0, 0))
    comparison.paste(mask, (test_image.width, 0))
    comparison.paste(output, (test_image.width * 2, 0))
    comparison.save('comparison_' + save_path)
    
    return output

def main():
    # Directory paths
    mutilated_dir = "data/digitized versions/Vies des saints/mutilations/"
    excised_dir = "data/digitized versions/Vies des saints/excisions/"
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # SD typically expects 512x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = ManuscriptDataset(mutilated_dir, excised_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Train model
    model_path = "runwayml/stable-diffusion-inpainting"  # or another suitable pre-trained model
    trained_pipe = train_inpainting_model(model_path, dataloader)
    
    # Save the fine-tuned model
    trained_pipe.save_pretrained("manuscript_restoration_model")
    
    # Test the model on a sample image
    test_image_path = os.path.join(mutilated_dir, "page_11_mutilated.jpeg")
    test_model(trained_pipe, test_image_path)

if __name__ == "__main__":
    main()