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

        # Apply mask to image to create mutilated image
        mutilated_image = image_transformed * mask

        # Create excision image by removing the mutilated parts
        excision_image = image_transformed * (1 - mask)

        return mutilated_image, excision_image, (1 - mask), img_name

    def generate_random_mask(self, height, width):
        mask = Image.new('L', (width, height), 1)  # Start with all ones (no masking)
        draw = ImageDraw.Draw(mask)

        # Define mask size (20% of the smallest dimension)
        mask_size = int(self.mask_ratio * min(height, width))
        top = random.randint(0, height - mask_size)
        left = random.randint(0, width - mask_size)
        bottom = top + mask_size
        right = left + mask_size

        # Draw rectangle mask (0 represents masked area)
        draw.rectangle([left, top, right, bottom], fill=0)

        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Shape: [1, H, W]
        return mask

def save_image(tensor, save_path):
    """
    Save a tensor as a PIL Image.

    tensor: Tensor of shape [3, H, W]
    save_path: Path to save the image
    """
    # Clamp the tensor to [0,1] range to avoid potential issues
    tensor = torch.clamp(tensor, 0, 1)

    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.cpu())

    # Save the image
    image.save(save_path)

def generate_and_save_mutilations(
    complete_dir='data/digitized versions/Vies des saints/jpeg/',
    mutilations_dir='data/digitized versions/Vies des saints/mutilations/',
    excisions_dir='data/digitized versions/Vies des saints/excisions/',
    masks_dir='data/digitized versions/Vies des saints/masks/',
    img_size=1000
):
    # Create output directories if they don't exist
    os.makedirs(mutilations_dir, exist_ok=True)
    os.makedirs(excisions_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset
    dataset = ManuscriptDataset(
        image_dir=complete_dir,
        transform=transform,
        mask_ratio=0.2
    )

    total = len(dataset)
    for idx in range(total):
        mutilated_img, excision_img, mask_img, img_name = dataset[idx]

        # Generate filenames for mutilated, excision, and mask images
        name, ext = os.path.splitext(img_name)
        mutilation_name = f"{name}_mutilated{ext}"
        excision_name = f"{name}_excised{ext}"
        mask_name = f"{name}_mask{ext}"

        # Define save paths
        mutilation_save_path = os.path.join(mutilations_dir, mutilation_name)
        excision_save_path = os.path.join(excisions_dir, excision_name)
        mask_save_path = os.path.join(masks_dir, mask_name)

        # Save the mutilated image
        save_image(mutilated_img, mutilation_save_path)

        # Save the excision image
        save_image(excision_img, excision_save_path)

        # Save the mask image
        save_image(mask_img, mask_save_path)

        # Optional: Print progress every 100 images
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {total} images")

    print(f"All mutilated images have been saved to {mutilations_dir}")
    print(f"All excised images have been saved to {excisions_dir}")
    print(f"All mask images have been saved to {masks_dir}")

if __name__ == "__main__":
    generate_and_save_mutilations()
