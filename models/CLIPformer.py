from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import clip
from networks.vision_transformer import Encoder
from einops import rearrange

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

        p = self.head(z).squeeze().permute(1, 0)
        return p, z
    
    def predict(
            self, 
            x: torch.Tensor,
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().max(-1).values.flatten().tolist()
        
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