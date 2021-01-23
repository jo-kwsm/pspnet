import torch.nn as nn

from .pspnet import PSPNet

__all__ = ["get_model"]


def get_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    model = PSPNet(n_classes=n_classes)

    # TODO 事前学習

    return model
