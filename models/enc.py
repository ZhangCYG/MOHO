import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34
import pytorch_lightning as pl


class ImageSpEnc(pl.LightningModule):
    def __init__(self, out_dim):
        super().__init__()

        model = resnet34(True)
        # 224 --> 56 -> 28 -> 14 -> 7
        self.net = model

        # dim = 256+512+1024+2048
        dim = 64+128+256+512
        self.z_head = nn.Conv2d(dim, out_dim, 1)
        self.global_head = nn.Sequential(nn.Conv2d(512, out_dim, 1), nn.Flatten())
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _foward_res(self, x):
        latent =[]

        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        latent.append(x)
        x = net.layer2(x)
        latent.append(x)
        x = net.layer3(x)
        latent.append(x)
        x = net.layer4(x)
        latent.append(x)
        x = net.avgpool(x)
        
        return latent, x

    def forward(self, x):
        latents, glb_latents = self._foward_res(x)
        align_corners = True
        latent_sz = latents[0].shape[-2:]
        for i, lat in enumerate(latents):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode='bilinear',
                align_corners=align_corners,
            )
        latents = torch.cat(latents, dim=1)
        latents = self.z_head(latents)
        glb_latents = self.global_head(glb_latents)

        return glb_latents, latents
