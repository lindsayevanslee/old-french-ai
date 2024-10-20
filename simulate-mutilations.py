import os
import random
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class ManuscriptDataset(Dataset):
    def __init__(self, image_dir, transform=None, mask_ratio=0.2):
        """
        image_dir: Directory with complete manuscript images.
        transform: Transformations to apply to the images.
        mask_ratio: Fraction of the image to mask.
        """
        self.image_dir = image_dir
        self.image_filenames = [
            fname for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.transform = transform
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = transforms.ToTensor()(image)

        # Create mask
        mask = self.generate_random_mask(image_transformed.size(1), image_transformed.size(2))

        # Apply mask to image
        masked_image = image_transformed * mask

        return masked_image, mask, image, img_name

    def generate_random_mask(self, height, width):
        mask = Image.new('L', (width, height), 1)
        draw = ImageDraw.Draw(mask)

        # Define mask size
        mask_size = int(self.mask_ratio * min(height, width))
        top = random.randint(0, height - mask_size)
        left = random.randint(0, width - mask_size)
        bottom = top + mask_size
        right = left + mask_size

        # Draw rectangle mask
        draw.rectangle([left, top, right, bottom], fill=0)
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Shape: [1, H, W]
        return mask

def save_mutiliation(masked_tensor, save_path):
    """
    Save the masked (mutilated) image to the specified path.
    
    masked_tensor: Tensor of shape [3, H, W]
    save_path: Path to save the masked image
    """
    # Clamp the tensor to [0,1] range
    masked_tensor = torch.clamp(masked_tensor, 0, 1)
    
    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    masked_image = to_pil(masked_tensor.cpu())

    # Save the image
    masked_image.save(save_path)

def generate_and_save_mutiliations(
    complete_dir='data/digitized versions/Vies des saints/jpeg/',
    mutiliations_dir='data/digitized versions/Vies des saints/mutiliations/',
    img_size=256
):
    os.makedirs(mutiliations_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = ManuscriptDataset(
        image_dir=complete_dir,
        transform=transform,
        mask_ratio=0.2
    )

    total = len(dataset)
    for idx in range(total):
        masked_img, mask, original_img, img_name = dataset[idx]

        # Generate a new filename
        name, ext = os.path.splitext(img_name)
        mutiliation_name = f"{name}_mutiliation{ext}"
        save_path = os.path.join(mutiliations_dir, mutiliation_name)

        # Save the masked image
        save_mutiliation(masked_img, save_path)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {total} images")

    print(f"All mutiliations have been saved to {mutiliations_dir}")

if __name__ == "__main__":
    generate_and_save_mutiliations()
