"""
Pretrain a Stable Diffusion model
"""
import os
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintTrainer, StableDiffusionInpaintDataset
from tqdm import tqdm
import PIL.Image
from transformers import AdamW, get_scheduler
from torchvision import transforms

# Load the inpainting pipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

# Base directory
base_dir = 'data/digitized versions/Vies des saints'

# Make directory for results
save_dir = f'{base_dir}/model_results/stablediffusion_trained'
os.makedirs(save_dir, exist_ok=True)

# Define prompt
prompt = "text of medieval manuscript"

# Loop through all images in the mutilations directory
mutilations_dir = f'{base_dir}/mutilations'
masks_dir = f'{base_dir}/masks'
excision_dir = f'{base_dir}/excisions'

# Prepare dataset
class ManuscriptInpaintingDataset(StableDiffusionInpaintDataset):
    def __init__(self, mutilations_dir, masks_dir, excision_dir, transform=None):
        self.mutilations_dir = mutilations_dir
        self.masks_dir = masks_dir
        self.excision_dir = excision_dir
        self.transform = transform
        self.image_files = os.listdir(mutilations_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mutilated_img = PIL.Image.open(os.path.join(self.mutilations_dir, img_name)).convert("RGB")
        mask_img = PIL.Image.open(os.path.join(self.masks_dir, img_name)).convert("L")
        excision_img = PIL.Image.open(os.path.join(self.excision_dir, img_name)).convert("RGB")

        if self.transform:
            mutilated_img = self.transform(mutilated_img)
            mask_img = self.transform(mask_img)
            excision_img = self.transform(excision_img)

        return mutilated_img, mask_img, excision_img

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = ManuscriptInpaintingDataset(mutilations_dir, masks_dir, excision_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Define optimizer and scheduler
num_epochs = 3
optimizer = AdamW(pipeline.parameters(), lr=5e-5)
num_training_steps = len(dataloader) * num_epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop

for epoch in range(num_epochs):
    pipeline.train()
    for batch in tqdm(dataloader):
        mutilated_imgs, masks, excision_imgs = batch
        mutilated_imgs = mutilated_imgs.to("cuda")
        masks = masks.to("cuda")
        excision_imgs = excision_imgs.to("cuda")

        outputs = pipeline(mutilated_imgs, masks, prompt=prompt)
        loss = torch.nn.functional.mse_loss(outputs, excision_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item()}")

# Save the trained model
pipeline.save_pretrained(save_dir)