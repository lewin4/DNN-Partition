import torch
import torch.nn as nn
from collections import OrderedDict
from Resnet_Model_Pair import *
from torchsummary import summary
from torchstat import stat
import numpy as np
from Branchy_Resnet18 import ResNet


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

    # net = NetExit4Part2R()
    # summary, _ = summary(net, (64, 48, 64), batch_size=2, device="cpu")
    # print("+"*20)
    # stat(net, (64, 48, 64))
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=6, branch=1)
    summary, _ = summary(model, (3, 192, 256), batch_size=2, device="cpu")
    print("+"*20)
    x = torch.rand(2, 3, 192, 256)
    y = model(x)
    stat(model, (3, 192, 256))
    print("Net ok!")
