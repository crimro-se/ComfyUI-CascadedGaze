import torch
import comfy
import comfy.utils
import comfy.model_management
from safetensors.torch import load_file
from safetensors import safe_open
import folder_paths
import json

from .cascadedgaze import CascadedGaze
from .tiling import process_image_batch

class DenoiseModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"), 
                              {"tooltip": "These models are loaded from the 'ComfyUI/models/upscale_models'"}),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
            }
        }
    
    RETURN_TYPES = ("DENOISE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "image/denoising"

    def load_model(self, model_name, dtype):
        model_path = folder_paths.get_full_path("upscale_models", model_name)

        # Load config
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()

        if not metadata or 'config' not in metadata or 'colorspace' not in metadata:
            raise ValueError("No configuration found in model metadata")

        config = json.loads(metadata['config'])

        # Initialize model
        model = CascadedGaze(**config)
        
        # Load weights
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
        
        # Setup device and dtype
        device = comfy.model_management.get_torch_device()
        model = model.to(device)
        
        # Handle dtype
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
            
        model = model.to(torch_dtype)
        model.requires_grad_(False)
        model.eval()
        
        # Store configuration in model object
        model.denoise_config = {
            "torch_dtype": torch_dtype,
            "device": device,
            "colorspace": metadata['colorspace']
        }
        
        return (model,)

class DenoiseImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("DENOISE_MODEL",),
                "image": ("IMAGE",),
                "overlap": ("INT", {"default": 12, "min": 4, "max": 64, "step": 4}),
                "patch_batch_size": ("INT", {"default": 12, "min": 1, "max": 64}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/denoising"

    def denoise(self, model, image, overlap, patch_batch_size):
        device = model.denoise_config["device"]
        torch_dtype = model.denoise_config["torch_dtype"]
        BGR_REQUIRED = model.denoise_config["colorspace"] == 'bgr'
        
        # Convert ComfyUI image to [B, C, H, W]
        image = image.permute(0, 3, 1, 2).to(device, dtype=torch_dtype)

        if BGR_REQUIRED:
            image = image[:,[2,1,0],:,:]

        # Process with denoiser
        with torch.inference_mode():
            denoised = process_image_batch(
                model=model,
                batch_images=image,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
                kernel_size=256
            )

        if BGR_REQUIRED:
            denoised = denoised[:,[2,1,0],:,:]

        # Convert back to ComfyUI format [B, H, W, C]
        denoised = denoised.permute(0, 2, 3, 1)
        
        # Ensure float32 output for ComfyUI
        return (denoised.to(torch.float32).clamp(0, 1),)