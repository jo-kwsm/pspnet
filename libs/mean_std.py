from typing import Tuple


def get_mean(norm_value: float = 255) -> Tuple[float]:
    return (0.485, 0.456, 0.406)


def get_std(norm_value: float = 255) -> Tuple[float]:
    return (0.229, 0.224, 0.225)
