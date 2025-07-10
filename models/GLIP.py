import torch
import torch.nn as nn
import clip
import numpy as np
from torchvision.transforms.functional import five_crop
from torchvision import transforms
from einops import rearrange
from typing import Union

def custom_unfold(img, crop_size, stride, normalize=True):
    """
    img: Tensor of shape [B, C, H, W]
    crop_size: tuple (h_crop, w_crop)
    stride: tuple (h_stride, w_stride)
    normalize: whether to apply CLIP-style normalization to each patch
    Returns: Tensor of shape [B, N_patches, C, h_crop, w_crop]
    """
    B, C, H, W = img.shape
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride

    h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1

    # Normalize transform
    normalize_fn = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ) if normalize else None

    all_patches = []

    for b in range(B):
        img_patches = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, H)
                x2 = min(x1 + w_crop, W)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                patch = img[b:b+1, :, y1:y2, x1:x2]  # Shape: [1, C, h_crop, w_crop]

                if normalize_fn is not None:
                    patch = normalize_fn(patch)

                img_patches.append(patch)

        # Stack patches for this image along a new dimension
        img_patches = torch.cat(img_patches, dim=0)  # Shape: [N_patches, C, h_crop, w_crop]
        all_patches.append(img_patches)

    # Stack all batches: [B, N_patches, C, h_crop, w_crop]
    all_patches = torch.stack(all_patches, dim=0)

    return all_patches

class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class GLIP(nn.Module):
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
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for param in self.clip.parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([257, 1, len(self.hooks), proj_dim])) # L_i x B x N x D_i
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
        self.proj1_cls = nn.Sequential(*proj1_layers)
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
        self.proj2_cls = nn.Sequential(*proj2_layers)
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
        self.head_cls = nn.Sequential(
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
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)
        
        g = self.proj1(g[1:, :, :, :].float())
        g_cls = self.proj1_cls(g[0, :, :, :].float())
        g = torch.cat([g_cls.unsqueeze(0), g], dim=0)

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z_cls = z[0, :, :]
        z = z[1:, :, :]
        
        z = self.proj2(z)
        z_cls = self.proj2_cls(z_cls)
        
        p_cls = self.head_cls(z_cls).squeeze()
        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z, p_cls, z_cls

    def forward_slide(self, img, stride=112, crop_size=224, batch_size_p=64, beta=0.5):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        imgs = custom_unfold(
            img,
            crop_size=crop_size,
            stride=stride,
        )
        logits = []
        for img in imgs:
            logits_img = []
            for i in range(0, img.shape[0], batch_size_p):
                batch_imgs = img[i:i + batch_size_p]
                logits_i, _, logits_o, _ = self.forward(batch_imgs)
                pred = logits_i.sigmoid().mean(-1) * beta + logits_o.sigmoid() * (1 - beta)
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                logits_img.append(pred)
            logits.append(torch.cat(logits_img, dim=0).mean())
        return torch.stack(logits, dim=0)
    

    def predict(self, img, **kwargs):
        with torch.no_grad():
            beta = kwargs.get("beta", 1.0)
            if kwargs.get('window_slide', False):
                stride = kwargs.get('stride', 112)
                if isinstance(img, list):
                    o = []
                    for i in img:
                        o_i = self.forward_slide(i, stride=stride, beta=beta)
                        o.append(o_i.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(img, stride=stride, beta=beta)
                    return o.squeeze().flatten().cpu().numpy()
            else:
                o_l, _, o_g, _ = self.forward(img)
                return beta * o_l.sigmoid().mean(-1).flatten().cpu().numpy() + (1 - beta) * o_g.sigmoid().flatten().cpu().numpy()
        
    def load_weights(self, ckpt: str):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt}")