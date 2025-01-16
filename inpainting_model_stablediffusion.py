import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
from torchvision import transforms
import numpy as np

class ManuscriptDataset(Dataset):
    def __init__(self, mutilated_dir, excised_dir, device='cuda'):
        self.mutilated_dir = mutilated_dir
        self.excised_dir = excised_dir
        self.device = device
        
        # Define image transforms for consistency
        self.image_transform = transforms.Compose([
            transforms.Resize(512),  # Stable Diffusion expects 512x512 images
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])
        
        # Simple transform for masks
        self.mask_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])
        
        # Get image pairs
        self.image_pairs = []
        for filename in os.listdir(mutilated_dir):
            if filename.endswith(('.jpeg', '.jpg', '.png')):
                excised_file = filename.replace('mutilated', 'excised')
                if os.path.exists(os.path.join(excised_dir, excised_file)):
                    self.image_pairs.append((filename, excised_file))
        
        print(f"Found {len(self.image_pairs)} image pairs")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # Load images
        mutilated_path = os.path.join(self.mutilated_dir, self.image_pairs[idx][0])
        excised_path = os.path.join(self.excised_dir, self.image_pairs[idx][1])
        
        mutilated = Image.open(mutilated_path).convert('RGB')
        excised = Image.open(excised_path).convert('RGB')
        
        # Apply transforms
        mutilated = self.image_transform(mutilated)
        excised = self.image_transform(excised)
        
        # Create binary mask from mutilated image
        # White (1) in areas to be inpainted, Black (0) elsewhere
        mask = torch.where(
            mutilated.mean(dim=0, keepdim=True) > 0.9,  # Threshold for white areas
            torch.ones(1, 512, 512),
            torch.zeros(1, 512, 512)
        )
        
        return mutilated, mask, excised

def train_inpainting_model(model_path, dataloader, device='cuda'):
    """Fine-tune the stable diffusion model for manuscript restoration"""
    print(f"Training on device: {device}")
    
    # Initialize model with consistent precision
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use full precision for stability
        safety_checker=None
    ).to(device)
    
    # Prepare text condition (empty string for unconditional generation)
    uncond_tokens = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    uncond_embeddings = pipe.text_encoder(uncond_tokens)[0]
    
    # Training configuration
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)
    
    for epoch in range(10):
        print(f"Starting epoch {epoch + 1}/10")
        for batch_idx, (image, mask, target) in enumerate(dataloader):
            # Move batch to device
            image = image.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            # Sample timesteps uniformly
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (image.shape[0],),
                device=device
            ).long()
            
            # Prepare latent variables
            with torch.no_grad():
                # Convert images to latent space
                latents = pipe.vae.encode(image).latent_dist.sample()
                latents = 0.18215 * latents  # Scale factor from VAE
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # Prepare masked image embeddings
                masked_image = image * (1 - mask)
                masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample() * 0.18215
                
                # Resize mask to match latent space dimensions
                mask = torch.nn.functional.interpolate(
                    mask,
                    size=latents.shape[-2:],
                    mode='nearest'
                )
            
            # Prepare UNet input by concatenating all components
            model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
            
            # Get UNet prediction
            noise_pred = pipe.unet(
                model_input,
                timesteps,
                encoder_hidden_states=uncond_embeddings
            ).sample
            
            # Calculate loss
            loss = nn.MSELoss()(noise_pred, noise)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Memory management
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
    
    return pipe

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define paths
    mutilated_dir = "data/digitized versions/Vies des saints/mutilations/"
    excised_dir = "data/digitized versions/Vies des saints/excisions/"
    
    # Create dataset and dataloader
    dataset = ManuscriptDataset(mutilated_dir, excised_dir, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Use batch size of 1 for stability
    
    # Train model
    model_path = "runwayml/stable-diffusion-inpainting"
    trained_pipe = train_inpainting_model(model_path, dataloader, device=device)
    
    # Save the trained model
    trained_pipe.save_pretrained("manuscript_restoration_model")

if __name__ == "__main__":
    main()