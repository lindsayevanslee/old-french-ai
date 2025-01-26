import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

def main():
    # Path to your newly saved inpainting model
    model_dir = "manuscript_restoration_model2"  # adjust if needed

    # Load the pipeline in half-precision
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # Load a test mutilated image
    test_mutilated_path = "data/digitized versions/Vies des saints/mutilations/page_11_mutilated.jpeg"
    mutilated_image = Image.open(test_mutilated_path).convert("RGB")

    # Load or create a corresponding mask
    # For the stable-diffusion inpainting pipeline, the mask is white in areas you want to inpaint, black where you do NOT want changes.
    test_mask_path = "data/digitized versions/Vies des saints/masks/page_11_mask.jpeg"
    mask_image = Image.open(test_mask_path).convert("RGB")

    # Optionally, you can provide a textual prompt.
    # For a purely “restorative” approach, you might leave it blank or try something minimal like "Medieval manuscript text".
    prompt = "Medieval manuscript text"

    # Perform the inpainting.
    # Use torch.autocast for speed and memory savings
    with torch.autocast("cuda"):
        result = pipe(prompt=prompt, image=mutilated_image, mask_image=mask_image).images[0]

    # Save or display
    output_path = "data/digitized versions/Vies des saints/model_results/stablediffusion_trained/restored_page.png"
    result.save(output_path)
    print(f"Saved restored image to {output_path}")

if __name__ == "__main__":
    main()
