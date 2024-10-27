import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from inpainting_model import UNetInpaint
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import gc
import torch.cuda

# Add this function
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


class ManuscriptInpaintingDataset(Dataset):
    def __init__(self, mutilations_dir, excisions_dir, transform=None):
        """
        Args:
            mutilations_dir (str): Directory with mutilated (mutilationed) images.
            excisions_dir (str): Directory with excised (removed) images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mutilations_dir = mutilations_dir
        self.excisions_dir = excisions_dir
        self.transform = transform
        
        # List of mutilated image filenames
        self.mutilation_filenames = [
            fname for fname in os.listdir(mutilations_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        # Ensure that for each mutilation, there's a corresponding excision
        self.mutilation_filenames = [
            fname for fname in self.mutilation_filenames
            if os.path.exists(os.path.join(excisions_dir, fname.replace('_mutilated', '_excised')))
        ]
        
    def __len__(self):
        return len(self.mutilation_filenames)
    
    def __getitem__(self, idx):
        mutilation_name = self.mutilation_filenames[idx]
        excision_name = mutilation_name.replace('_mutilated', '_excised')
        
        mutilation_path = os.path.join(self.mutilations_dir, mutilation_name)
        excision_path = os.path.join(self.excisions_dir, excision_name)
        
        # Load images
        mutilation_image = Image.open(mutilation_path).convert('RGB')
        excision_image = Image.open(excision_path).convert('RGB')
        
        if self.transform:
            mutilation_image = self.transform(mutilation_image)
            excision_image = self.transform(excision_image)
        
        # Create mask from excision image
        # Assuming excision_image has zeros where no excision and non-zeros where excision
        # Convert to grayscale and threshold
        excision_gray = excision_image.mean(dim=0)  # [H, W]
        mask = (excision_gray > 0).float().unsqueeze(0)  # [1, H, W]
        
        # Input is mutilated image + mask
        input_image = torch.cat([mutilation_image, mask], dim=0)  # [4, H, W]
        
        # Target is original image: mutilation + excision
        target_image = mutilation_image + excision_image  # [3, H, W]
        target_image = torch.clamp(target_image, 0, 1)  # Ensure pixel values are in [0,1]
        
        return input_image, target_image

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    save_every=10,
    model_save_path='models/unet_inpaint.pth'
):
    """
    Trains the model.
    
    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on ('cuda' or 'cpu').
        num_epochs (int): Number of epochs to train.
        save_every (int): Save the model every 'save_every' epochs.
        model_save_path (str): Path to save the trained model.
    """
    model.to(device)
    
    for epoch in range(num_epochs):
        clear_gpu_memory()  # Clear memory at start of epoch

        # Training Phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)    # [B, 4, H, W]
            targets = targets.to(device)  # [B, 3, H, W]
            
            # Forward pass
            outputs = model(inputs)       # [B, 3, H, W]
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'Loss': loss.item()})

            # Clear memory after each batch
            del outputs, loss
            clear_gpu_memory()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Val)"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_running_loss += loss.item() * inputs.size(0)
                
                # Convert tensors to numpy arrays for metric calculation
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                for i in range(outputs_np.shape[0]):
                    # PSNR and SSIM expect HWC and range [0,1]
                    output_img = outputs_np[i].transpose(1, 2, 0)
                    target_img = targets_np[i].transpose(1, 2, 0)
                    
                    # Calculate PSNR
                    psnr_val = psnr(target_img, output_img, data_range=1)
                    
                    # Calculate SSIM with explicit parameters
                    ssim_val = ssim(
                        target_img, 
                        output_img,
                        win_size=5,  # Smaller window size
                        channel_axis=2,  # Specify channel axis
                        data_range=1
                    )
                    
                    psnr_total += psnr_val
                    ssim_total += ssim_val
        
        val_loss = val_running_loss / len(val_loader.dataset)
        avg_psnr = psnr_total / len(val_loader.dataset)
        avg_ssim = ssim_total / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        
        # Save the model every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    
    # Save the final model
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    # Paths to directories
    mutilations_dir = 'data/digitized versions/Vies des saints/mutilations/'
    excisions_dir = 'data/digitized versions/Vies des saints/excisions/'
    
    # Hyperparameters
    num_epochs = 50
    batch_size = 1          # Reduced batch size due to larger image size
    learning_rate = 1e-4
    img_size = 1000         # Updated image size
    mask_ratio = 0.2        # Should match data_preparation.py
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Initialize Dataset
    dataset = ManuscriptInpaintingDataset(
        mutilations_dir=mutilations_dir,
        excisions_dir=excisions_dir,
        transform=transform
    )
    
    # Split dataset into training and validation (e.g., 80% train, 20% val)
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    # Initialize the model
    model = UNetInpaint(in_channels=4, out_channels=3)
    
    # Define Loss and Optimizer
    criterion = nn.L1Loss()  # L1 Loss is effective for image inpainting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Start Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_every=10,
        model_save_path='models/unet_inpaint.pth'
    )
