import comfy
from .nodes import DenoiseModelLoader, DenoiseImage


# Node mappings
NODE_CLASS_MAPPINGS = {
    "CGDenoiseModelLoader": DenoiseModelLoader,
    "CGDenoiseImage": DenoiseImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CGDenoiseModelLoader": "Load CascadedGaze Denoise Model",
    "CGDenoiseImage": "Apply CascadedGaze Image Denoising"
}