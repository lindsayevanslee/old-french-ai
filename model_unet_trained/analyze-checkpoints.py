import torch
import os

def analyze_checkpoints(checkpoint_dir='models'):
    # Find all checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('unet_inpaint_epoch_')]
    
    best_metrics = {
        'psnr': float('-inf'),
        'ssim': float('-inf'),
        'val_loss': float('inf'),
        'best_psnr_epoch': 0,
        'best_ssim_epoch': 0,
        'best_val_loss_epoch': 0
    }
    
    # Analyze each checkpoint
    for checkpoint_file in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        
        epoch = checkpoint['epoch']
        
        # Update best PSNR
        if checkpoint['psnr'] > best_metrics['psnr']:
            best_metrics['psnr'] = checkpoint['psnr']
            best_metrics['best_psnr_epoch'] = epoch + 1
            
        # Update best SSIM
        if checkpoint['ssim'] > best_metrics['ssim']:
            best_metrics['ssim'] = checkpoint['ssim']
            best_metrics['best_ssim_epoch'] = epoch + 1
            
        # Update best validation loss
        if checkpoint['val_loss'] < best_metrics['val_loss']:
            best_metrics['val_loss'] = checkpoint['val_loss']
            best_metrics['best_val_loss_epoch'] = epoch + 1
    
    print("\nBest Model Metrics:")
    print(f"Best PSNR: {best_metrics['psnr']:.2f} (Epoch {best_metrics['best_psnr_epoch']})")
    print(f"Best SSIM: {best_metrics['ssim']:.4f} (Epoch {best_metrics['best_ssim_epoch']})")
    print(f"Best Validation Loss: {best_metrics['val_loss']:.4f} (Epoch {best_metrics['best_val_loss_epoch']})")
    
    return best_metrics

if __name__ == "__main__":
    best_metrics = analyze_checkpoints()