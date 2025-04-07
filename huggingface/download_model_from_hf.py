from huggingface_hub import hf_hub_download, login
from getpass import getpass
import os

def download_model(username, repo_name="old-french-ai"):
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model file
    model_path = hf_hub_download(
        repo_id=f"{username}/{repo_name}",
        filename="unet_inpaint_epoch_40.pth",
        local_dir="models"
    )
    
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    # Login to Hugging Face
    username = "lindsayevanslee"

    token = getpass("Enter your Hugging Face token: ")  # Get from https://huggingface.co/settings/tokens
    login(token)

    #Download model
    download_model(username)