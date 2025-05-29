import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained

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
    
    def predict(
            self, 
            x: torch.Tensor,
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().mean(-1).flatten().cpu().numpy()
            # return o.sigmoid().max(-1).values.flatten().tolist()
            
    def load_weights(self, ckpt: str):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt}")