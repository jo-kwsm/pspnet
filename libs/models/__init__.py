import os

import torch
import torch.nn as nn

from .pspnet import PSPNet

__all__ = ["get_model"]


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)


def get_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    model = PSPNet(n_classes=150)

    if pretrained:
        weights_dir = "./libs/models/weights"
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        pspnet_weights_path = os.path.join(weights_dir, "pspnet50_ADE20K.pth")

        pspnet_weights = torch.load(pspnet_weights_path)
        model.load_state_dict(pspnet_weights)

    model.decode_feature.classification = nn.Conv2d(
    in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    model.aux.classification = nn.Conv2d(
    in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    model.decode_feature.classification.apply(weights_init)
    model.aux.classification.apply(weights_init)

    return model
