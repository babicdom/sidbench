import torch
from typing import Iterable
import numpy as np

# PSM
def patch_collate(batch):
    try:
        # input_img = torch.stack([item[0] for item in batch], dim=0)
        input_img = [item[0] for item in batch]
        cropped_img = torch.stack([item[1] for item in batch], dim=0)
        scale = torch.stack([item[2] for item in batch], dim=0)
        target = torch.tensor([item[3] for item in batch])
        filename=[item[4] for item in batch]
        return [input_img, cropped_img, scale], target, filename
    except Exception as e:
        raise ValueError('Error in patch_collate: ' + str(e))
    
# SPAI
def image_enlisting_collate_fn(
    batch: Iterable[tuple[torch.Tensor, np.ndarray, int]]
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate function that enlists its entries."""
    return (
        [torch.utils.data.default_collate([s[0]]) for s in batch],
        torch.utils.data.default_collate([s[1] for s in batch]),
        [s[2] for s in batch]
    )
