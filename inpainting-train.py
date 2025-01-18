"""
Memory-efficient training script for manuscript inpainting model.
Optimized for large images while maintaining training stability.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from inpainting_model import UNetInpaint
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import gc
import torch.cuda
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    torch.cuda.empty_cache()

class ManuscriptInpaintingDataset(Dataset):
    """Dataset class for loading manuscript images with pre-computed masks."""
    def __init__(self, mutilations_dir, excisions_dir, masks_dir, transform=None):
        self.mutilations_dir = Path(mutilations_dir)
        self.excisions_dir = Path(excisions_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Get all mutilated image files
        self.image_files = [
            f for f in self.mutilations_dir.glob("*.jpeg")
            if f.stem.endswith('_mutilated')
        ]
        
        # Verify matching files exist
        self.image_files = [
            f for f in self.image_files
            if self._has_matching_files(f)
        ]
        
        logging.info(f"Found {len(self.image_files)} valid image sets")
        
    def _has_matching_files(self, mutilated_file):
        """Check if all required matching files exist."""
        base_name = mutilated_file.stem.replace('_mutilated', '')
        excised_file = self.excisions_dir / f"{base_name}_excised.jpeg"
        mask_file = self.masks_dir / f"{base_name}_mask.jpeg"
        return excised_file.exists() and mask_file.exists()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get file paths
        mutilated_path = self.image_files[idx]
        base_name = mutilated_path.stem.replace('_mutilated', '')
        excised_path = self.excisions_dir / f"{base_name}_excised.jpeg"
        mask_path = self.masks_dir / f"{base_name}_mask.jpeg"
        
        # Load and resize images before converting to tensors
        mutilated_img = Image.open(mutilated_path).convert('RGB')
        excised_img = Image.open(excised_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        
        if self.transform:
            mutilated_img = self.transform(mutilated_img)
            excised_img = self.transform(excised_img)
            mask_img = self.transform(mask_img)
        
        # Ensure mask is binary
        mask_img = (mask_img > 0.5).float()
        
        # Combine inputs
        input_tensor = torch.cat([mutilated_img, mask_img], dim=0)
        
        return input_tensor, excised_img, mask_img

class InpaintingLoss(nn.Module):
    """Simple L1 loss with mask weighting."""
    def __init__(self, alpha=0.7):
        super(InpaintingLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred, target, mask):
        # Regular L1 loss
        l1_loss = torch.abs(pred - target).mean()
        
        # Masked L1 loss
        masked_loss = (torch.abs(pred - target) * mask).mean()
        
        return self.alpha * masked_loss + (1 - self.alpha) * l1_loss

def validate_model(model, val_loader, criterion, device):
    """Run validation loop with memory optimization and robust metric calculation.
    
    This function handles validation carefully by:
    1. Computing metrics on full images first
    2. Using a smaller window size for SSIM calculation
    3. Properly handling edge cases in metric computation
    """
    model.eval()
    val_loss = 0
    val_metrics = {'psnr': 0, 'ssim': 0}
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Generate predictions
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            val_loss += loss.item()
            
            # Move tensors to CPU and convert to numpy for metric calculation
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(outputs_np.shape[0]):
                # Convert images to HWC format for metric calculation
                output_img = np.transpose(outputs_np[i], (1, 2, 0))
                target_img = np.transpose(targets_np[i], (1, 2, 0))
                
                # Ensure values are in valid range
                output_img = np.clip(output_img, 0, 1)
                target_img = np.clip(target_img, 0, 1)
                
                try:
                    # Calculate PSNR
                    psnr_value = psnr(target_img, output_img, data_range=1.0)
                    
                    # Calculate SSIM with smaller window size and explicit channel axis
                    ssim_value = ssim(
                        target_img, 
                        output_img,
                        win_size=5,  # Use smaller window size
                        channel_axis=2,  # Specify channel axis
                        data_range=1.0,
                        gaussian_weights=True,  # Use Gaussian weighting for better stability
                    )
                    
                    # Update metrics only if calculation was successful
                    if not np.isnan(psnr_value):
                        val_metrics['psnr'] += psnr_value
                    if not np.isnan(ssim_value):
                        val_metrics['ssim'] += ssim_value
                        
                except Exception as e:
                    # Log error but continue validation
                    logging.warning(f"Error calculating metrics: {str(e)}")
                    continue
            
            num_batches += 1
            
            # Clear GPU memory
            del outputs, loss
            clear_gpu_memory()
    
    # Calculate averages, handling case where some metrics failed
    val_loss /= max(num_batches, 1)  # Avoid division by zero
    
    for metric in val_metrics:
        val_metrics[metric] /= max(num_batches, 1)
        # Log if metrics seem suspicious
        if val_metrics[metric] < 0 or val_metrics[metric] > 100:
            logging.warning(f"Suspicious {metric} value: {val_metrics[metric]}")
    
    return val_loss, val_metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=50, save_every=5, checkpoint_dir='checkpoints'):
    """Memory-efficient training loop."""
    model.to(device)
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Look for latest checkpoint
    checkpoints = list(Path(checkpoint_dir).glob('unet_inpaint_epoch_*.pth'))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[3]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logging.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for inputs, targets, masks in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            progress_bar.set_postfix({'Loss': loss.item()})
            
            # Clear GPU memory
            del outputs, loss
            clear_gpu_memory()
        
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        val_loss, val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Log metrics
        logging.info(f"""Epoch {epoch+1}/{num_epochs}:
            Train Loss: {avg_train_loss:.4f}
            Val Loss: {val_loss:.4f}
            PSNR: {val_metrics['psnr']:.2f}
            SSIM: {val_metrics['ssim']:.4f}
        """)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'metrics': val_metrics
            }, os.path.join(checkpoint_dir, 'unet_inpaint_best.pth'))
            logging.info("Saved best model checkpoint")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'metrics': val_metrics
            }, os.path.join(checkpoint_dir, f'unet_inpaint_epoch_{epoch+1}.pth'))
            logging.info(f"Saved checkpoint for epoch {epoch+1}")

def main():
    """Main training function with memory-optimized parameters."""
    # Paths
    mutilations_dir = 'data/digitized versions/Vies des saints/mutilations/'
    excisions_dir = 'data/digitized versions/Vies des saints/excisions/'
    masks_dir = 'data/digitized versions/Vies des saints/masks/'
    checkpoint_dir = 'checkpoints'
    
    # Training parameters - optimized for memory efficiency
    batch_size = 1  # Keep batch size at 1 for large images
    num_epochs = 50
    learning_rate = 1e-4
    img_size = 1000  # Maintain original size
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = ManuscriptInpaintingDataset(
        mutilations_dir=mutilations_dir,
        excisions_dir=excisions_dir,
        masks_dir=masks_dir,
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # Create data loaders with memory-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model, loss, and optimizer
    model = UNetInpaint(in_channels=4, out_channels=3)
    criterion = InpaintingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_every=5,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == "__main__":
    main()