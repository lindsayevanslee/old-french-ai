import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


"""
Testing SAM 2 on a sample image
Installation instructions: https://github.com/facebookresearch/sam2?tab=readme-ov-file
Clone the sam2 repo in the same parent directory as this repo, not within this repo

Example notebook: https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb

Automatic mask generation function with documentation about the parameters: https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py
"""


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

#load sample image
image = Image.open('data/page_13.jpeg')
image = np.array(image.convert("RGB"))

#paths to model checkpoints and configuration
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

#generate masks
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(sam2)

mask_generator = SAM2AutomaticMaskGenerator(
    model = sam2,
    points_per_side=48,           # Increased from default 32 for finer sampling
    points_per_batch = 16,
    pred_iou_thresh=0.7,          # Lowered from 0.8 to be more permissive
    stability_score_thresh=0.85,  # Lowered from 0.95 to detect more features
    stability_score_offset=0.8,   # Slightly lower than default
    crop_n_layers=1,              # Add 1 layer of cropping for detail
    box_nms_thresh=0.6,           # Lowered to reduce overlapping masks
    min_mask_region_area=400,     # Set minimum area to avoid tiny segments
)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())


plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
output_path = "data/page_13_sam2_masks.png"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Annotated image saved to {output_path}")


all_masks = [
    mask['segmentation']
    for mask
    in sorted(masks, key=lambda x: x['area'], reverse=True)
]

# Save all masks as images
for i, mask in enumerate(all_masks):
    output_mask_path = f"data/page_13_sam2_mask_{i + 1}.png"
    plt.imsave(output_mask_path, mask, cmap='gray')
    print(f"Mask {i + 1} saved to {output_mask_path}")
