import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained
from typing import Union
from torchvision import transforms
import numpy as np

class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class SigLIPIntermediate(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.siglip, self.preprocess = create_model_from_pretrained(backbone[0], device=device) # 'hf-hub:timm/ViT-L-16-SigLIP2-256', device=device)
        for name, param in self.siglip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.siglip.visual.named_modules()
            if "ls2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([256, 1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            self.siglip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)
        g = g.permute(1, 0, 2, 3)
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z
    
    def forward_slide(self, img, stride=128, crop_size=256, patch_size=16, reshape=True):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        assert stride % patch_size == 0, f"Stride muste be divisible by patch size ({patch_size})"
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        n_h, n_w = h_img // patch_size, w_img // patch_size
        s_h, s_w = h_stride // patch_size, w_stride // patch_size
        h_img, w_img = n_h * patch_size, n_w * patch_size
        h_w, w_w = h_crop // patch_size, w_crop // patch_size

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = img.new_zeros((batch_size, n_h, n_w))
        count_mat = img.new_zeros((batch_size, n_h, n_w))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                h_1, w_1 = h_idx * s_h, w_idx * s_w
                h_2, w_2 = min(h_1 + h_w, n_h), min(w_1 + w_w, n_w)
                h_1, w_1 = max(h_2 - h_w, 0), max(w_2 - w_w, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img = transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                )(crop_img)
                crop_seg_logit, _ = self.forward(crop_img)
                crop_seg_logit = crop_seg_logit.reshape(-1, h_w, w_w)

                preds += nn.functional.pad(crop_seg_logit,
                               (int(w_1), int(preds.shape[2] - w_2), int(h_1),
                                int(preds.shape[1] - h_2)))

                count_mat[:, h_1:h_2, w_1:w_2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat

        if reshape:
            return preds.reshape(batch_size, -1)
        else:
            return preds

    def predict(
            self, 
            x: Union[torch.Tensor, list[torch.Tensor]],
            **kwargs
    ):
        with torch.no_grad():
            p = kwargs.get("p", 1)
            method = kwargs.get("method", "mean")
            if kwargs.get("window_slide", False):
                stride = kwargs.get("stride", 128)
                if isinstance(x, list):
                    o = []
                    for xi in x: 
                        o_i = self.forward_slide(xi, stride=stride)
                        if method == "mean":
                            o.append(o_i.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy())
                        elif method == "max":
                            o.append(o_i.sigmoid().max(-1).values.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(x, stride=stride)
                    if method == "mean":
                        return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                    elif method == "max":
                        return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            else:
                o, _ = self.forward(x)
                if method == "mean":
                    return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                elif method == "max":
                    return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            
    def load_weights(self, ckpt: str):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt}")