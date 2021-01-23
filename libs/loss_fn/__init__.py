import torch.nn as nn

from libs.loss_fn.psploss import PSPLoss

__all__ = ["get_criterion"]

def get_criterion(
    aux_weight: float=0.4,
) -> nn.Module:
    criterion = PSPLoss(aux_weight=0.4)

    return criterion
