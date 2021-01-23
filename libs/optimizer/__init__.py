import math
from typing import Any, Tuple

import torch.nn as nn
import torch.optim as optim

__all__ = ["get_optimizer"]


def get_optimizer(model: nn.Module) -> Tuple[Any, Any]:
    optimizer = optim.SGD([
        {'params': model.feature_conv.parameters(), 'lr': 1e-3},
        {'params': model.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': model.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': model.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': model.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': model.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': model.decode_feature.parameters(), 'lr': 1e-2},
        {'params': model.aux.parameters(), 'lr': 1e-2},
    ], momentum=0.9, weight_decay=0.0001)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    return optimizer, scheduler

def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow((1-epoch/max_epoch), 0.9)
