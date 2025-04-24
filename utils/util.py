import os
import torch
from collections import OrderedDict
from typing import Optional, Iterable
from PIL import Image
import pathlib
from matplotlib import colormaps

import numpy as np
import random


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape N x C x H x W 
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(mean)[None, :, None, None]


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def setup_device(gpus):
    if gpus == '-1':  # CPU
        device = torch.device('cpu')
    else:
        device_ids = list(map(int, gpus.split(',')))
        if len(device_ids) == 1:
            # Single GPU
            device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
        else:
            # Multiple GPUs (DataParallel)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("CUDA is not available. Running on CPU.")
            else:
                torch.cuda.set_device(device_ids[0])  # Set default device for DataParallel
    return device

# From SPAI
def save_image_with_attention_overlay(
    image_patches: torch.Tensor,
    attention_scores: list[float],
    original_height: int,
    original_width: int,
    patch_size: int,
    stride: int,
    overlayed_image_path: pathlib.Path,
    alpha: float = 0.4,
    palette: str = "coolwarm",
    mask_path: Optional[pathlib.Path] = None,
    overlay_path: Optional[pathlib.Path] = None
) -> None:
    """Overlays attention scores over image patches."""
    out_height_blocks: int = ((original_height - patch_size) // stride) + 1
    out_width_blocks: int = ((original_width - patch_size) // stride) + 1
    out_height: int = ((out_height_blocks - 1) * stride) + patch_size
    out_width: int = ((out_width_blocks - 1) * stride) + patch_size

    attention_scores: np.ndarray = np.array(attention_scores)
    # Normalize attention scores in [0, 1].
    # attention_scores = (attention_scores - attention_scores.min()) / attention_scores.max()

    cmap = colormaps[palette]
    # cmap_attention_scores: np.ndarray = cmap(np.array(attention_scores))[:, :3]

    out_image: np.ndarray = np.ones((out_height, out_width, 3))
    overlay: np.ndarray = np.zeros((out_height, out_width))
    overlay_count: np.ndarray = np.zeros_like(overlay)
    for i in range(out_height_blocks):
        for j in range(out_width_blocks):
            out_image[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size,
                :
            ] = image_patches[0, (i*out_width_blocks)+j].detach().cpu().permute((1, 2, 0)).numpy()
            overlay[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size
            ] += attention_scores[(i*out_width_blocks)+j]
            overlay_count[
                i * stride:i * stride + patch_size,
                j * stride:j * stride + patch_size
            ] += 1
    overlay = overlay / overlay_count
    overlay = (overlay - overlay.min()) / overlay.max()

    colormapped_overlay = cmap(overlay)[:, :, :3]

    overlayed_image: np.ndarray = (1 - alpha) * out_image + alpha * colormapped_overlay
    overlayed_image = (overlayed_image * 255).astype(np.uint8)

    Image.fromarray(overlayed_image).save(overlayed_image_path)
    if mask_path is not None:
        Image.fromarray((overlay*255).astype(np.uint8)).save(mask_path)
    if overlay_path is not None:
        Image.fromarray((colormapped_overlay*255).astype(np.uint8)).save(overlay_path)
