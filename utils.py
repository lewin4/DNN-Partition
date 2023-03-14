import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
from config import IMAGE_DIM
import numpy as np
import torch
import os


def test_data():
    testtransform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    testdata = datasets.CIFAR10(root='./CIFAR', train=False, transform=testtransform, download=True)

    valdataloader = data.DataLoader(
        testdata,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=1)

    dataiter = iter(valdataloader)
    return dataiter


def load_regression_data(server_regression_result_dir: str):
    data = {}
    for regression_type_name in [name for name in os.listdir(server_regression_result_dir) if
                                 name.split(".")[-1] == "pth"]:
        type_regression = torch.load(server_regression_result_dir + regression_type_name,
                                     map_location=torch.device('cpu'))

        weight = type_regression["weight"]
        cat_weight = np.empty(0, dtype=np.float64)
        for key, value in weight.items():
            value = torch.squeeze(value).numpy()
            cat_weight = np.append(cat_weight, value)
        type_regression["weight"] = list(cat_weight)

        x_max = type_regression["x_max"]
        x_max = torch.squeeze(x_max, 0).numpy()
        x_max = list(x_max)
        type_regression["x_max"] = x_max

        x_min = type_regression["x_min"]
        x_min = list(torch.squeeze(x_min, 0).numpy())
        type_regression["x_min"] = x_min

        y_max = type_regression["y_max"]
        y_max = list(torch.squeeze(y_max, 0).numpy())
        type_regression["y_max"] = y_max

        y_min = type_regression["y_min"]
        y_min = list(torch.squeeze(y_min, 0).numpy())
        type_regression["y_min"] = y_min

        data[regression_type_name.split(".")[0]] = type_regression

    return data
