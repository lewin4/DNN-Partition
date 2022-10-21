import torch
import torch.nn as nn
from collections import OrderedDict
from Resnet_Model_Pair import *
from torchsummary import summary
from torchstat import stat
import numpy as np


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
    x = torch.tensor([[2.0, 1.0]])
    x2 = torch.tensor([[4.0, 3.0]])
    x3 = x * x2
    fc = torch.nn.Linear(2, 1)
    fc.weight = torch.nn.Parameter(torch.tensor([[3.0, 4.0]]))
    fc.bias = torch.nn.Parameter(torch.tensor([5.0]))
    y = fc(x)
    print("Net ok!")
