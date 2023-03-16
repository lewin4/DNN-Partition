"""
小脚本
把归一化后的权重和偏置变成没有归一化后的权重和偏置
"""
from regression import load_train_data
from config import regression_type
import torch

if "__main__" == __name__:
    torch.set_printoptions(precision=16)
    for type in regression_type:
        checkpoint = load_train_data("regression_output/regression_result/{}.pth".format(type))
        weight = checkpoint["weight"]
        max_data_x = checkpoint["x_max"]
        min_data_x = checkpoint["x_min"]
        max_data_y = checkpoint["y_max"]
        min_data_y = checkpoint["y_min"]

        W = (max_data_y - min_data_y)/(max_data_x - min_data_x)*(weight["liner.weight"].cpu())
        B = (max_data_y - min_data_y)*(weight["liner.bias"].cpu()) + min_data_y - \
            (max_data_y - min_data_y)*torch.sum((weight["liner.weight"].cpu())*min_data_x/(max_data_x - min_data_x))
        print(type)
        print("W:")
        print(W)
        print("B:")
        print(B)
        print("====================================================")