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
                 device='cuda'):
        self.mutilated_dir = mutilated_dir
        self.excised_dir = excised_dir
        self.device = device
        
        # Image transform: downsize to 512×512, normalize to [-1,1]
        self.image_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        # Gather image pairs
        self.image_pairs = []
        for filename in os.listdir(mutilated_dir):
            if filename.endswith(('.jpeg', '.jpg', '.png')):
                excised_file = filename.replace('mutilated', 'excised')
                if os.path.exists(os.path.join(excised_dir, excised_file)):
                    self.image_pairs.append((filename, excised_file))
        
        print(f"Found {len(self.image_pairs)} image pairs in dataset.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        mutilated_path = os.path.join(self.mutilated_dir, self.image_pairs[idx][0])
        excised_path   = os.path.join(self.excised_dir, self.image_pairs[idx][1])
        
        # Load images
        mutilated = Image.open(mutilated_path).convert('RGB')
        excised   = Image.open(excised_path).convert('RGB')
        
        # Transforms
        mutilated_t = self.image_transform(mutilated)
        excised_t   = self.image_transform(excised)
        
        # Example: threshold-based mask
        mask_t = generate_mask_threshold(mutilated_t, threshold=0.9)
        
        # Return all three. We only train using the mutilated image & mask,
        # but we might want the excised for debugging/inspection.
        return mutilated_t, mask_t, excised_t

def train_inpainting_model(model_path, dataloader, device='cuda'):
    """Fine-tune the stable diffusion inpainting model with 
       VAE and text encoder frozen, in half-precision, 
       using gradient checkpointing & attention slicing."""
    print(f"Training on device: {device}")

    ############################################################################
    # 1) Load pipeline in half precision. Then freeze text encoder + VAE.
    ############################################################################
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # half precision
        safety_checker=None
    )
    pipe.to(device)
    
    # Freeze text encoder + VAE (no gradient needed)
    pipe.text_encoder.eval()
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    
    pipe.vae.eval()
    for param in pipe.vae.parameters():
        param.requires_grad = False
    
    # Ensure the UNet is in train mode
    pipe.unet.train()

    # Enable memory-saving features
    pipe.unet.enable_gradient_checkpointing()
    pipe.enable_attention_slicing()

    # Optionally use a simpler scheduler with fewer steps
    pipe.scheduler = DDPMScheduler(
        num_train_timesteps=250,  # fewer steps -> less memory/time
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    ############################################################################
    # 2) Precompute unconditional text embedding (we won't train text encoder)
    ############################################################################
    with torch.no_grad():
        uncond_tokens = pipe.tokenizer(
            [""],  # empty prompt
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        uncond_embeddings = pipe.text_encoder(uncond_tokens)[0]
        # Just in case, detach (not strictly needed if text encoder is eval/frozen):
        uncond_embeddings = uncond_embeddings.detach()

    ############################################################################
    # 3) Set up optimizer (training only the UNet)
    ############################################################################
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)

    # Scale factor from the original stable diffusion config
    vae_scaling_factor = 0.18215

    ############################################################################
    # 4) Training loop
    ############################################################################
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (image, mask, excised) in enumerate(dataloader):
            # Move data to GPU in half precision
            image = image.half().to(device)
            mask  = mask.half().to(device)

            # ---------------------------------------------------
            # 4A) Convert images to latents with VAE, no grad
            # ---------------------------------------------------
            with torch.no_grad():
                latents = pipe.vae.encode(image).latent_dist.sample()
                latents = latents * vae_scaling_factor
                # Detach from the graph because VAE is frozen
                latents = latents.detach()

                # Generate random noise
                noise = torch.randn_like(latents)

                # Pick random timesteps
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                ).long()

                # Add noise to latents
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents.detach()

                # Compute masked image latents
                masked_image = image * (1 - mask)
                masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae_scaling_factor
                masked_image_latents = masked_image_latents.detach()

                # Resize mask to match the latent resolution
                mask_resized = torch.nn.functional.interpolate(
                    mask, size=latents.shape[-2:], mode='nearest'
                )
                mask_resized = mask_resized.detach()

            # ---------------------------------------------------
            # 4B) Forward pass in UNet (the only part we train)
            # ---------------------------------------------------
            # Input for inpainting typically concatenates [noisy_latents, mask, masked_image_latents]
            model_input = torch.cat([noisy_latents, mask_resized, masked_image_latents], dim=1)

            # The UNet forward pass
            noise_pred = pipe.unet(
                model_input,
                timesteps,
                encoder_hidden_states=uncond_embeddings
            ).sample  # shape [B, 4, H, W] typically

            # ---------------------------------------------------
            # 4C) Compute loss: MSE between predicted noise and actual noise
            # ---------------------------------------------------
            loss = nn.MSELoss()(noise_pred, noise)

            # ---------------------------------------------------
            # 4D) Backprop + step
            # ---------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Clean up GPU mem occasionally
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

    return pipe

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define paths
    mutilated_dir = "data/digitized versions/Vies des saints/mutilations/"
    excised_dir   = "data/digitized versions/Vies des saints/excisions/"

    # Create dataset and dataloader
    dataset = ManuscriptDataset(
        mutilated_dir=mutilated_dir,
        excised_dir=excised_dir,
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Train model
    model_path = "runwayml/stable-diffusion-inpainting"
    trained_pipe = train_inpainting_model(model_path, dataloader, device=device)

    # Save the trained model
    trained_pipe.save_pretrained("manuscript_restoration_model")
    print("Done! Model saved to 'manuscript_restoration_model'")

if __name__ == "__main__":
    main()
