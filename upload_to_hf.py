from huggingface_hub import HfApi, login
from getpass import getpass
import os

# Login to Hugging Face
my_username = "lindsayevanslee"

token = getpass("Enter your Hugging Face token: ")  # Get from https://huggingface.co/settings/tokens
login(token)

api = HfApi()

# Create a new repository
repo_name = "old-french-ai" 
api.create_repo(repo_name, private=True) 

# Upload the model file
model_path = "models/unet_inpaint.pth"
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="unet_inpaint.pth",
    repo_id=f"{my_username}/{repo_name}", 
    commit_message="Upload trained model weights"
)

# Create improved README with metadata
readme_content = """---
language: en
tags:
- pytorch
- inpainting
- manuscript-restoration
- image-restoration
- unet
library_name: pytorch
pipeline_tag: image-to-image
#datasets:
#- custom manuscript dataset
license: mit
---

# Manuscript Inpainting Model

This model performs inpainting on historical manuscript images to restore damaged or missing sections.

## Model Description

- **Model Architecture:** UNet with double convolution blocks
- **Task:** Image inpainting/restoration
- **Domain:** Historical manuscripts
- **Input:** RGB image + binary mask (4 channels)
- **Output:** Restored RGB image (3 channels)
- **Model Size:** ~124MB

### Training Data

The model was trained on:
- Historical manuscript images from [describe your dataset]
- Image size: 512x512 pixels
- Number of training images: [number]
- Types of damage: [describe types of damage/restoration needed]

### Training Procedure

- **Framework:** PyTorch
- **Training Duration:** 50 epochs
- **Optimization:** Adam optimizer
- **Loss Function:** L1 Loss
- **Evaluation Metrics:** PSNR, SSIM
- **Hardware:** NVIDIA T4 GPU on Google Cloud

## Performance

The model achieved:
- PSNR: [your best PSNR value]
- SSIM: [your best SSIM value]
- Validation Loss: [your best validation loss]

## Usage

```python
import torch
from inpainting_model import UNetInpaint
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="lindsayevanslee/manuscript-inpainting",
    filename="unet_inpaint.pth"
)

# Load the model
model = UNetInpaint()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

# Prepare your input (example)
# input_tensor should be: [batch_size, 4, height, width]
# where the 4 channels are: RGB + mask
output = model(input_tensor)
```
"""



# Upload model card (documentation)
with open("README.md", "w") as f:
    f.write(readme_content)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=f"{my_username}/{repo_name}",
    commit_message="Add model documentation"
)
print(f"Model uploaded to https://huggingface.co/{my_username}/{repo_name}")
            