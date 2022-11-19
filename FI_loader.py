import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
from torch.utils.data.dataloader import DataLoader
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from typing import List
import torchvision.transforms as transforms


class SewageDataset(Dataset):
    def __init__(self, image_dir, radio=None, mode: str = "train", transform=None, seed=10086):
        if radio is None:
            radio = [0.75, 0.25]
        self.image_dir = image_dir
        self.split = "/" if os.name == "posix" else "\\"
        self.transform = transform
        self.images = []
        classes = os.listdir(self.image_dir)
        for ficlass in classes:
            class_path = os.path.join(self.image_dir, ficlass)
            for image in [x for x in os.listdir(class_path) if x.endswith(".jpg") or x.endswith(".jpeg")]:
                self.images.append(os.path.join(class_path, image))
        random.seed(seed)
        random.shuffle(self.images)
        self.len = len(self.images)
        assert mode in ["train", "val", "test"], "mode should is one of [train, val, test]"
        if mode == "train":
            self.images = self.images[:int(radio[0] * self.len)]
        elif mode == "val":
            self.images = self.images[int(radio[0] * self.len):]
        elif mode == "test":
            self.transform = transforms.Compose(
                [
                    # A.Resize(height=192, width=256),
                    transforms.Resize((192, 256)),
                    # A.Normalize(
                    #     mean=[0.5330, 0.5463, 0.5493],
                    #     std=[0.1143, 0.1125, 0.1007],
                    #     max_pixel_value=255.0,
                    # ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5330, 0.5463, 0.5493],
                                         std=[0.1143, 0.1125, 0.1007],),
                ],)
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_path = self.images[item]
        image = Image.open(img_path)
        # image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)
            # image = augmentations["image"]
        else:
            raise ValueError("Transformer is None.")

        mask = torch.Tensor([int(img_path.split(self.split)[-2]),])

        return image, mask


def get_loaders(image_dir: str,
                batch_size: int,
                img_shape: List[int],
                num_workers=0,
                pin_memory=True,
                radio=None,
                train_transform=None,
                val_transform=None,
                ):
    if radio is None:
        radio = [0.75, 0.25]
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_shape[0], img_shape[1])),
                transforms.RandomRotation(35),
                # A.Rotate(limit=35, p=1.0),
                # A.HorizontalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.1),
                transforms.RandomVerticalFlip(p=0.1),
                # A.Normalize(
                #     mean=[0.5330, 0.5463, 0.5493],
                #     std=[0.1143, 0.1125, 0.1007],
                #     max_pixel_value=255.0,
                # ),
                # ToTensorV2(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5330, 0.5463, 0.5493],
                                     std=[0.1143, 0.1125, 0.1007], ),
            ],
        )
    if val_transform is None:
        val_transform = transforms.Compose(
            [
                transforms.Resize((img_shape[0], img_shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5330, 0.5463, 0.5493],
                                     std=[0.1143, 0.1125, 0.1007], ),
            ],
        )
    train_dataset = SewageDataset(image_dir, radio=radio, mode="train", transform=train_transform)
    val_dataset = SewageDataset(image_dir, radio=radio, mode="val", transform=val_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True,
                              pin_memory=pin_memory,
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size,
                            shuffle=True,
                            pin_memory=pin_memory,
                            num_workers=num_workers)

    return train_loader, val_loader
