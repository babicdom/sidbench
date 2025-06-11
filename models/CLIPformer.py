from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import numpy as np
import clip
from networks.vision_transformer import Encoder
from einops import rearrange
import pickle
from typing import Union
from torchvision import transforms

CLIP_SEQ_LENGTH=256

class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class IntermediatePatch(nn.Module):
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
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
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
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[1:, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z
    
    def forward_slide(self, img, stride=112, crop_size=224, patch_size=14, reshape=True):
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
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
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
                stride = kwargs.get("stride", 112)
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

class PatchAttention(nn.Module):
    def __init__(
            self, 
            att_dim: int,
            n_heads: int,
            hidden_dim: int,
            dropout: int = 0.0,
        ):
        super().__init__()
        dim_head: int = att_dim // n_heads
        self.heads = n_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.k = nn.Linear(hidden_dim, att_dim, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros((n_heads, 1, att_dim//n_heads)))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)

    def forward(
            self, 
            x: torch.Tensor,
    ):
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        k = self.k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        if self.heads > 1:
            attn = attn.mean(dim=1)
        attn = attn.squeeze()
        return attn

class AttentionIntermediatePatch(nn.Module):
    def __init__(
        self,
        att_dim,
        n_heads,
        device,
    ):
        super().__init__()

        self.device = device

        opt = pickle.load(
            open(f"weights/IntermediatePatch/experiment_progan.pickle", "rb")
        )
        self.intermediate_patch = IntermediatePatch(
            backbone=opt["backbone"],
            nproj=opt["nproj"],
            proj_dim=opt["proj_dim"],
            device=torch.device("cuda:0"),
        )
        self.intermediate_patch.load_state_dict(
            torch.load(f"weights/IntermediatePatch/train_progan.pth", map_location="cuda:0")
        )

        for name, param in self.intermediate_patch.named_parameters():
            param.requires_grad = False

        self.window_attention = PatchAttention(
            att_dim=att_dim,
            n_heads=n_heads,
            hidden_dim=opt["proj_dim"],
        )
        
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            p, z = self.intermediate_patch(x)
        g = self.window_attention(z.permute(1, 0, 2))
        p = p * g
        p = p.sum(dim=1)
        return p, g
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().flatten().cpu().numpy()

    def load_weights(self, ckpt: str):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt}")

class PatchAttentionPool(nn.Module):
    def __init__(
            self, 
            att_dim: int,
            n_heads: int,
            dropout: int,
            hidden_dim: int,
        ):
        super().__init__()
        dim_head: int = att_dim // n_heads
        self.heads = n_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.kv = nn.Linear(hidden_dim, att_dim*2, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros((n_heads, 1, att_dim//n_heads)))
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)
        self.o = nn.Sequential(
            nn.Linear(att_dim, hidden_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
            self, 
            x: torch.Tensor,
            return_attn: bool = False,
    ):
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        kv = self.kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.o(x)
        x = x.squeeze(dim=1)
        if return_attn:
            return x, attn
        else:
            return x


class CLIPformer(nn.Module):
    def __init__(
        self,
        backbone,
        device,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        att_dim: int,
        num_classes: int = 1,
        cls_ration: int = 1,
        cls_dropout: float = 0.5,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hook to get the last layer tokens
        self.hook = Hook("transformer.resblocks.23.ln_2", self.clip.visual.transformer.resblocks[-1].ln_2)

        # Extension
        hidden_dim = backbone[1]
        self.encoder = Encoder(
            seq_length=CLIP_SEQ_LENGTH,
            num_layers=n_layers,
            num_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # Patch Attention Pooling
        self.patch_attention_pool = PatchAttentionPool(
            att_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )

        # Classification head
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*cls_ration),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ration, hidden_dim*cls_ration),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ration, num_classes)
        )
        
    def forward(
            self, 
            x: torch.Tensor
    ):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = self.hook.output[1:, :, :]
        g = g.permute(1, 0, 2)
        g = self.encoder(g)
        g = self.patch_attention_pool(g)
        o = self.cls(g).squeeze(-1)
        return o, g
    
    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().flatten().tolist()
    
    def load_weights(self, ckpt: str):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt}")

class CLIPatch(nn.Module):
    def __init__(
        self,
        backbone,
        device,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        att_dim: int,
        num_classes: int = 1,
        cls_ratio: int = 1,
        cls_dropout: float = 0.5,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hook to get the last layer tokens
        self.hook = Hook("transformer.resblocks.23.ln_2", self.clip.visual.transformer.resblocks[-1].ln_2)

        # Extension
        hidden_dim = backbone[1]
        self.encoder = Encoder(
            seq_length=CLIP_SEQ_LENGTH,
            num_layers=n_layers,
            num_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # Patch Attention
        self.patch_attention = PatchAttentionPool(
            att_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )

        # Classification head
        self.num_classes = num_classes
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*cls_ratio),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ratio, hidden_dim*cls_ratio),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ratio, num_classes)
        )
        
    def forward(
            self, 
            x: torch.Tensor
    ):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = self.hook.output[1:, :, :]
        g = g.permute(1, 0, 2)
        g = self.encoder(g)
        attn = self.patch_attention(g)

        batch_size, num_patches, embedding_dim = g.shape
        g_reshaped = g.reshape(-1, embedding_dim).float()
        out_flat = self.cls(g_reshaped)
        
        if self.num_classes == 1:
            out = out_flat.reshape(batch_size, num_patches)
        else:
            out = out_flat.reshape(batch_size, num_patches, self.num_classes) 
        
        return out, attn, g
    
    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            o, _, _ = self.forward(x)
            # return o.sigmoid().mean(-1).flatten().tolist()
            return o.sigmoid().max(-1).values.flatten().tolist()
    
    def load_weights(self, ckpt: str):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt}")