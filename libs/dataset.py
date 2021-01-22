import os, sys
from typing import Any, Optional

import pandas as pd
from PIL import Image
import numpy as np
import torch
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_dataloader(
    csv_file: str,
    phase: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:

    data = VOCDataset(
        csv_file,
        phase,
        transform=transform,
    )
    
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class VOCDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        phase: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        assert os.path.exists(csv_file)

        csv_path = os.path.join(csv_file)

        self.df = pd.read_csv(csv_path)
        self.phase = phase
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        img, anno_class_img = self.pull_item(idx)
        return img, anno_class_img

    def pull_item(self, idx: int) -> Any:
        image_file_path = self.df.iloc[idx]["image_path"]
        segmentation_file_path = self.df.iloc[idx]["segmentation_path"]

        img = Image.open(image_file_path)
        anno_class_img = Image.open(segmentation_file_path)

        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


def data_test():
    print(sys.path)
    import numpy as np
    import matplotlib.pyplot as plt
    from transformer import DataTransform
    
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    input_size = 475
    train_path = "csv/train.csv"
    val_path = "csv/val.csv"

    train_dataset = VOCDataset(train_path, phase="train", transform=DataTransform(
    input_size, color_mean, color_std))

    val_dataset = VOCDataset(val_path, phase="val", transform=DataTransform(
    input_size, color_mean, color_std))

    print(val_dataset.__getitem__(1))

    train_loader = get_dataloader(
        csv_file=train_path,
        phase="train",
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        transform=DataTransform(input_size, color_mean, color_std),
    )

    val_loader = get_dataloader(
        csv_file=val_path,
        phase="val",
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        transform=DataTransform(input_size, color_mean, color_std),
    )

    batch_iterator = iter(train_loader)
    imgs, anno_class_imgs = next(batch_iterator)

    print(imgs.size())

    val_img = imgs[0].numpy().transpose((1, 2, 0))
    plt.imshow(val_img)
    plt.show()


if __name__ == "__main__":
    data_test()
