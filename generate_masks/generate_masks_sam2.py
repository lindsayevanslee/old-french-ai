import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Default configuration name - change this to use a different configuration
DEFAULT_CONFIG_NAME = "toulouse_page120"

"""
Testing SAM 2 on a sample image
Installation instructions: https://github.com/facebookresearch/sam2?tab=readme-ov-file
Clone the sam2 repo in the same parent directory as this repo, not within this repo

Example notebook: https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb

Automatic mask generation function with documentation about the parameters: https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py
"""

def load_config(config_path, config_name):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    for conf in config['model_inputs']:
        if conf['name'] == config_name:
            return conf
    
    raise ValueError(f"Configuration '{config_name}' not found in {config_path}")

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

def main():
    parser = argparse.ArgumentParser(description='Generate masks using SAM2')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--config-name', type=str, default=DEFAULT_CONFIG_NAME,
                        help=f'Name of the configuration to use (default: {DEFAULT_CONFIG_NAME})')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.config_name)

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
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    #load sample image
    image = Image.open(config['input_image_path'])
    image = np.array(image.convert("RGB"))

    #paths to model checkpoints and configuration
    sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    #generate masks
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model = sam2,
        points_per_side=32,           # Reduced to focus on larger features
        points_per_batch = 16,
        pred_iou_thresh=0.6,          # Increased to be more selective
        stability_score_thresh=0.8,   # Increased to be more selective
        stability_score_offset=0.7,   # Increased to be more selective
        crop_n_layers=1,
        box_nms_thresh=0.7,           # Increased to reduce overlapping masks
        min_mask_region_area=5000,    # Significantly increased to focus on large regions only
    )

    masks = mask_generator.generate(image)

    print(len(masks))
    print(masks[0].keys())

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')

    # Get the directory path from input image
    output_dir = os.path.dirname(config['input_image_path'])
    base_filename = os.path.splitext(os.path.basename(config['input_image_path']))[0]

    # Save the annotated image with masks
    output_path = os.path.join(output_dir, f"{base_filename}_sam2_masks.png")
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
        output_mask_path = os.path.join(output_dir, f"{base_filename}_sam2_mask_{i + 1}.png")
        plt.imsave(output_mask_path, mask, cmap='gray')
        print(f"Mask {i + 1} saved to {output_mask_path}")

if __name__ == "__main__":
    main()
