import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

from networks.clip import clip 


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class RineModel(nn.Module):
    def __init__(self, backbone, nproj, proj_dim):
        super(RineModel, self).__init__()

        # Load and freeze CLIP
        self.clip, _ = clip.load(backbone[0], device="cpu")
        for _, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module) for name, module in self.clip.visual.named_modules() if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))

        proj1_layers = [
            nn.Dropout()
        ]

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

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return p, z

    def forward_slide(self, img, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_img, w_img = img.shape[-2:]
        h_crop, w_crop = crop_size, crop_size
        h_stride, w_stride = stride, stride

        if h_crop > h_img or w_crop > w_img:
            h_crop, w_crop = min(h_crop, h_img), min(w_crop, w_img)
            h_stride, w_stride = 1, 1

        patches = []
        for i in range(0, h_img - h_crop + 1, h_stride):
            for j in range(0, w_img - w_crop + 1, w_stride):
                patch = img[:, :, i:i + h_crop, j:j + w_crop]
                patch = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])(patch)
                patches.append(patch)

        patches = torch.concat(patches, dim=0)
        logits, _ = self.forward(patches)

        return logits.sigmoid().mean(dim=0).view(-1, 1)
    

    def predict(self, img, **kwargs):
        with torch.no_grad():
            if kwargs.get('window_slide', False):
                stride = kwargs.get('stride', 112)
                if isinstance(img, list):
                    o = []
                    for i in img:
                        o_i = self.forward_slide(i, stride=stride)
                        o.append(o_i.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(img, stride=stride)
                    return o.squeeze().flatten().cpu().numpy()
            else:
                logits, _ = self.forward(img)
                return logits.sigmoid().flatten().tolist()
        
    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        # for name in state_dict:
        #     exec(f'self.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])')
        self.load_state_dict(state_dict, strict=False)