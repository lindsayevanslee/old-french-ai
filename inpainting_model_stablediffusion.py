import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from PIL import Image
import os
from torchvision import transforms

##########################
# Example Mask Generation
##########################
def generate_mask_threshold(mutilated_tensor, threshold=0.9):
    """
    Expects a 3D tensor: CxHxW with values in [-1,1].
    We'll convert that range to [0,1] for thresholding.
    Returns a 2D mask [1, H, W] with 1 where area is "white," else 0.
    """
    # Convert from [-1,1] to [0,1]
    tensor_01 = (mutilated_tensor * 0.5) + 0.5
    mean_vals = tensor_01.mean(dim=0, keepdim=True)
    mask = torch.where(mean_vals > threshold, 1.0, 0.0)
    return mask

class ManuscriptDataset(Dataset):
    def __init__(self,
                 mutilated_dir,
                 excised_dir,
                 masks_dir,          # New parameter for masks directory
                 device='cuda'):
        self.mutilated_dir = mutilated_dir
        self.excised_dir = excised_dir
        self.masks_dir = masks_dir   # Store masks directory
        self.device = device
        
        # Image transform: downsize to 512×512, normalize to [-1,1]
        self.image_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5])
        ])
        
        # Mask transform: downsize to 512×512, convert to tensor
        # Note: No normalization for masks - they should stay as 0s and 1s
        self.mask_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])

        # Gather image triplets (mutilated, excised, mask)
        self.image_triplets = []
        for filename in os.listdir(mutilated_dir):
            if filename.endswith(('.jpeg', '.jpg', '.png')):
                # Get corresponding excised and mask filenames
                excised_file = filename.replace('mutilated', 'excised')
                mask_file = filename.replace('mutilated', 'mask')
                
                # Check if both corresponding files exist
                if (os.path.exists(os.path.join(excised_dir, excised_file)) and 
                    os.path.exists(os.path.join(masks_dir, mask_file))):
                    self.image_triplets.append((filename, excised_file, mask_file))
        
        print(f"Found {len(self.image_triplets)} complete image sets in dataset.")

    def __len__(self):
        return len(self.image_triplets)

    def __getitem__(self, idx):
        mutilated_path = os.path.join(self.mutilated_dir, self.image_triplets[idx][0])
        excised_path = os.path.join(self.excised_dir, self.image_triplets[idx][1])
        mask_path = os.path.join(self.masks_dir, self.image_triplets[idx][2])
        
        # Load all three components
        mutilated = Image.open(mutilated_path).convert('RGB')
        excised = Image.open(excised_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load mask as grayscale
        
        # Apply transforms
        mutilated_t = self.image_transform(mutilated)
        excised_t = self.image_transform(excised)
        mask_t = self.mask_transform(mask)
        
        # Ensure mask is binary (0 or 1)
        mask_t = (mask_t > 0.5).float()
        
        # For SD inpainting, mask should be 1 where we want to inpaint
        # If your masks are inverted (1 where content exists), uncomment:
        # mask_t = 1 - mask_t
        
        return mutilated_t, mask_t, excised_t

def train_inpainting_model(model_path, dataloader, device='cuda'):
    """Fine-tune the stable diffusion inpainting model with correct precision handling."""
    print(f"Training on device: {device}")

    # Initialize the pipeline in full precision first
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Start with full precision
        safety_checker=None
    )
    pipe.to(device)
    
    # Freeze text encoder + VAE
    pipe.text_encoder.eval()
    pipe.vae.eval()
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    
    # Prepare UNet for training
    pipe.unet.train()
    pipe.unet.enable_gradient_checkpointing()
    pipe.enable_attention_slicing(slice_size="auto")
    
    # Use simpler scheduler
    pipe.scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )

    # Create conditioning prompt
    text_prompt = ["restore this medieval manuscript text"]
    
    # Precompute text embeddings
    with torch.no_grad():
        tokens = pipe.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        
        text_embeddings = pipe.text_encoder(tokens)[0]
        text_embeddings = text_embeddings.detach()

    # Initialize optimizer with stability settings
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler()  # Updated from torch.cuda.amp
    
    vae_scaling_factor = 0.18215

    num_epochs = 10
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        for batch_idx, (mutilated, mask, excised) in enumerate(dataloader):
            # Move data to device (keep in full precision)
            mutilated = mutilated.to(device)
            mask = mask.to(device)
            excised = excised.to(device)

            optimizer.zero_grad()

            # Use automatic mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    # Encode images
                    mutilated_latents = pipe.vae.encode(mutilated).latent_dist.sample()
                    mutilated_latents = mutilated_latents * vae_scaling_factor
                    
                    # Generate noise
                    noise = torch.randn_like(mutilated_latents)
                    timestep = torch.randint(
                        0,
                        pipe.scheduler.config.num_train_timesteps,
                        (1,),
                        device=device
                    ).long()
                    
                    # Add noise
                    noisy_latents = pipe.scheduler.add_noise(
                        mutilated_latents,
                        noise,
                        timestep.repeat(mutilated_latents.shape[0])
                    )

                    # Prepare masked image latents
                    masked_image = mutilated * (1 - mask)
                    masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample()
                    masked_image_latents = masked_image_latents * vae_scaling_factor

                    # Resize mask
                    mask_resized = torch.nn.functional.interpolate(
                        mask,
                        size=mutilated_latents.shape[-2:],
                        mode='nearest'
                    )

                # Prepare model input
                model_input = torch.cat([noisy_latents, mask_resized, masked_image_latents], dim=1)

                # Forward pass
                noise_pred = pipe.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=text_embeddings.repeat(model_input.shape[0], 1, 1)
                ).sample

                # Compute loss
                loss = nn.MSELoss()(noise_pred.float(), noise.float())

            # Scale loss and compute gradients
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            if not torch.isnan(loss):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_grad_norm)
            
                # Update weights
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Clear GPU memory periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # Update learning rate based on epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # Early stopping if loss is NaN
        if torch.isnan(torch.tensor(avg_epoch_loss)):
            print("Training stopped due to NaN loss")
            break

    # Convert back to half precision for saving
    pipe.to(torch_dtype=torch.float16)
    return pipe

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define paths
    mutilated_dir = "data/digitized versions/Vies des saints/mutilations/"
    excised_dir = "data/digitized versions/Vies des saints/excisions/"
    masks_dir = "data/digitized versions/Vies des saints/masks/"  # New path

    # Create dataset and dataloader with masks directory
    dataset = ManuscriptDataset(
        mutilated_dir=mutilated_dir,
        excised_dir=excised_dir,
        masks_dir=masks_dir,     # Add masks directory
        device=device
    )
    
    # Using a larger batch size since we're not computing masks on the fly
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Train model
    model_path = "runwayml/stable-diffusion-inpainting"
    trained_pipe = train_inpainting_model(model_path, dataloader, device=device)

    # Save the trained model
    trained_pipe.save_pretrained("manuscript_restoration_model3")
    print("Done! Model saved to 'manuscript_restoration_model3'")

if __name__ == "__main__":
    main()
