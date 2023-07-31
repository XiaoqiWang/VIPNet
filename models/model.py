import torch
from torch import nn
from torchvision import models

from . import resnet
from .BoTNet import botnet
from .transformer import TransformerEncoder

class VIPNet(nn.Module):
    def __init__(self, config):
        super(VIPNet, self).__init__()
        # define content perception module
        if config.multi_scale:  # Using multi-scale features
            self.cpm = resnet.__dict__[config.cpm](config.cpm_channels, config.is_freeze_cpm)
        else:  # Using the features of the final network layer
            model = models.__dict__[config.cpm](pretrained=True)
            self.cpm = torch.nn.Sequential(*list(model.children())[:-2])
            if config.is_freeze_cpm:
                for p in self.cpm.parameters():
                    p.requires_grad = False
        # define distortion perception module
        dpm = botnet(config.dpm_checkpoints, resolution=(288, 384), heads=16, num_classes=150)
        self.dpm = torch.nn.Sequential(*list(dpm.children())[:-2])
        if config.is_freeze_dpm:
            for p in self.dpm.parameters():
                p.requires_grad = False
        # the output dimensions of the two modules
        cpm_dims = 512 if (config.cpm == 'resnet18' or config.cpm == 'resnet34') else 2048
        cpm_dims = config.cpm_channels if config.multi_scale else cpm_dims
        dpm_dims = 2048
        # define visual interaction module
        self.vim = TransformerEncoder(image_size=(9,12), channels=(dpm_dims + cpm_dims),patch_size=1, dim=config.embed_dim,
                                      depth=config.depth, heads=config.num_heads, mlp_dim=config.embed_dim*4, dropout=0.4, emb_dropout=0.4)
        # prediction of image quality score
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, 1)

    def forward(self, rgb, ycbcr):

        f_content = self.cpm(rgb)
        f_distortion = self.dpm(ycbcr)

        v = self.vim(f_content, f_distortion)
        q = self.head(self.norm(v)[:, 0])

        return q.squeeze()


