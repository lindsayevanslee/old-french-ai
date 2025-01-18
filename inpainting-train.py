"""
Training script for manuscript inpainting model that utilizes pre-computed masks
and implements advanced training procedures with comprehensive metrics.
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
        
        # Load images
        mutilated_img = Image.open(mutilated_path).convert('RGB')
        excised_img = Image.open(excised_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')  # Load as grayscale
        
        # Apply transformations
        if self.transform:
            mutilated_img = self.transform(mutilated_img)
            excised_img = self.transform(excised_img)
            mask_img = self.transform(mask_img)
        
        # Ensure mask is binary
        mask_img = (mask_img > 0.5).float()
        
        # Combine mutilated image and mask
        input_tensor = torch.cat([mutilated_img, mask_img], dim=0)
        
        # Target is the excised content
        target_tensor = excised_img
        
        return input_tensor, target_tensor, mask_img

class InpaintingLoss(nn.Module):
    """Custom loss function combining L1 loss with perceptual mask-weighted loss."""
    def __init__(self, alpha=0.7):
        super(InpaintingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        
    def forward(self, pred, target, mask):
        # Regular L1 loss
        l1_loss = self.l1(pred, target)
        
        # Masked L1 loss (focusing on inpainting region)
        masked_loss = self.l1(pred * mask, target * mask)
        
        # Combine losses
        return self.alpha * masked_loss + (1 - self.alpha) * l1_loss

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    save_every=5,
    checkpoint_dir='checkpoints'
):
    """
    Train the inpainting model with comprehensive logging and validation.
    """
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
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        clear_gpu_memory()
        model.train()
        epoch_loss = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for batch_idx, (inputs, targets, masks) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
            
            # Clear batch from GPU
            del outputs, loss
            clear_gpu_memory()
        
        avg_train_loss = epoch_loss / len(train_loader)


        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'psnr': 0, 'ssim': 0, 'masked_psnr': 0, 'masked_ssim': 0}
        num_val_samples = 0
        
        with torch.no_grad():
            for inputs, targets, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                for i in range(outputs_np.shape[0]):
                    # Convert to correct format for metrics (HWC)
                    output_img = np.transpose(outputs_np[i], (1, 2, 0))
                    target_img = np.transpose(targets_np[i], (1, 2, 0))
                    mask_img = np.transpose(masks_np[i], (1, 2, 0))
                    
                    # Calculate metrics for full image
                    val_metrics['psnr'] += psnr(target_img, output_img, data_range=1.0)
                    val_metrics['ssim'] += ssim(
                        target_img, 
                        output_img,
                        multichannel=True,
                        data_range=1.0
                    )
                    
                    # Calculate metrics for masked region only
                    masked_output = output_img * mask_img
                    masked_target = target_img * mask_img
                    val_metrics['masked_psnr'] += psnr(masked_target, masked_output, data_range=1.0)
                    val_metrics['masked_ssim'] += ssim(
                        masked_target,
                        masked_output,
                        multichannel=True,
                        data_range=1.0
                    )
                    
                num_val_samples += outputs_np.shape[0]
                
                # Clear batch from GPU
                del outputs, loss
                clear_gpu_memory()
        
        # Calculate average metrics
        avg_val_loss = val_loss / len(val_loader)
        for metric in val_metrics:
            val_metrics[metric] /= num_val_samples
        
        # Log metrics
        logging.info(f"""Epoch {epoch+1}/{num_epochs}:
            Train Loss: {avg_train_loss:.4f}
            Val Loss: {avg_val_loss:.4f}
            PSNR: {val_metrics['psnr']:.2f}
            SSIM: {val_metrics['ssim']:.4f}
            Masked PSNR: {val_metrics['masked_psnr']:.2f}
            Masked SSIM: {val_metrics['masked_ssim']:.4f}
        """)
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'unet_inpaint_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': val_metrics
            }, checkpoint_path)
            logging.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'unet_inpaint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': val_metrics
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    # Paths
    mutilations_dir = 'data/digitized versions/Vies des saints/mutilations/'
    excisions_dir = 'data/digitized versions/Vies des saints/excisions/'
    masks_dir = 'data/digitized versions/Vies des saints/masks/'
    checkpoint_dir = 'checkpoints'
    
    # Training parameters
    batch_size = 1  # Keep batch size small due to image size
    num_epochs = 50
    learning_rate = 1e-4
    img_size = 1000
    
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
    
    # Create data loaders
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