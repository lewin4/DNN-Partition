import torch
import torch.nn as nn
from collections import OrderedDict
from Resnet_Model_Pair import *
from torchsummary import summary
from torchstat import stat
import numpy as np
from Branchy_Resnet18 import ResNet
from config import *


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# name  = "alexnet_data_out/models/epoch_1_model.pt"
# state_dict = torch.load(name, map_location=device)
# model = nn.Sequential(OrderedDict([
#     ("conv1", nn.Conv2d(64, 32, 3, padding=1, stride=1)),
#     ("relu", nn.ReLU())
# ]))
# model_state_dict = model.state_dict()
# print(state_dict)

if __name__ == "__main__":

    # for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):    #iter[2, 1, 0]
    #     partition_point_number = [2, 2, 3, 4]
    #     for partition_point in range(partition_point_number[exit_branch]):
    #         L_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'L'
    #         R_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'R'

    aa = torch.nn.Conv2d(5, 64, (3, 3), 2)
    param = aa.weight
    stride = aa.stride
    print(aa)

    # model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=6, branch=1)
    # summary, _ = summary(model, (3, 192, 256), batch_size=2, device="cpu")
    # print("+"*20)
    # x = torch.rand(2, 3, 192, 256)
    # y = model(x)
    # stat(model, (3, 192, 256))
    # print("Net ok!")
