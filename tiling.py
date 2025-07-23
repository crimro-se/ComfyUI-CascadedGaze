import torch
import torch.nn.functional as F


def process_image_batch(model, batch_images, overlap=32, patch_batch_size=32, device='cuda', kernel_size=256):
    """
    Process batch of images using patching with overlap.
    
    Args:
        model: Pre-trained PyTorch model (input: [B,3,256,256], output: [B,C,256,256])
        batch_images: Input images as tensor [N,3,H,W] (float, values in [0,1])
            important note: only N=1 has been tested.
        overlap: Overlap between patches (pixels)
        patch_batch_size: Batch size for patch processing
        device: Target device for computation
        
    Returns:
        Processed images [N,C,H,W]
    """
    if overlap >= kernel_size:
        raise ValueError(f"Overlap must be smaller than kernel_size. Got {overlap} >= {kernel_size}")

    model.to(device).eval()
    N, _, H_orig, W_orig = batch_images.shape
    stride = kernel_size - overlap
    output_dtype = batch_images.dtype

    # Calculate padding
    H_pad = ((H_orig - 1) // stride) * stride + kernel_size
    W_pad = ((W_orig - 1) // stride) * stride + kernel_size
    pad_h = max(0, H_pad - H_orig)
    pad_w = max(0, W_pad - W_orig)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    pad_left, pad_right, pad_top, pad_bottom = padding

    # Precompute divisor mask (same for all images)
    ones = torch.ones(1, 1, H_pad, W_pad, dtype=output_dtype, device=device)
    ones_unfold = F.unfold(ones, kernel_size=kernel_size, stride=stride)
    divisor = F.fold(ones_unfold, (H_pad, W_pad), kernel_size, stride=stride)
    del ones, ones_unfold  # Free intermediate memory

    # Process each image sequentially
    output_list = []
    for i in range(N):
        # Process one image at a time
        img = batch_images[i:i+1]
        
        # Pad and move to device
        img_padded = F.pad(img, padding, mode='reflect').to(device)
        
        # Unfold into patches [1, C, H, W] -> [L, 3, kernel_size, kernel_size]
        patches = F.unfold(img_padded, kernel_size, stride=stride)
        L = patches.size(-1)
        patches = patches.permute(0, 2, 1).reshape(-1, 3, kernel_size, kernel_size)
        del img_padded  # Free memory immediately after use
        
        # Process patches in batches
        processed = []
        for j in range(0, len(patches), patch_batch_size):
            with torch.no_grad():
                batch = patches[j:j+patch_batch_size]
                out = model(batch)
                processed.append(out)
        processed_patches = torch.cat(processed, dim=0)
        del patches  
        
        # Reshape for folding [L, C, k, k] -> [1, C*k*k, L]
        patches_to_fold = processed_patches.reshape(1, L, -1).permute(0, 2, 1)
        
        # Fold and normalize
        folded = F.fold(
            patches_to_fold,
            output_size=(H_pad, W_pad),
            kernel_size=kernel_size,
            stride=stride,
        ) / divisor

        del patches_to_fold
        
        # Crop and store
        cropped = folded[..., pad_top:H_pad - pad_bottom, pad_left:W_pad - pad_right]
        output_list.append(cropped.clamp(0,1))
    
    return torch.cat(output_list, dim=0)