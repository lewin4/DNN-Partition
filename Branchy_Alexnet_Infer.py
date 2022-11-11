import torch
import numpy as np
from Resnet_Model_Pair import *
from config import *
from collections import OrderedDict


def load_model_param(model: torch.nn.Module, path: str):
    model_state_dict = model.state_dict()
    param_state_dict = torch.load(path)

    assert set(model_state_dict.keys()).issubset(set(param_state_dict.keys()))
    model_state_dict = OrderedDict({key: param_state_dict[key] for key in model_state_dict.keys()})
    model.load_state_dict(model_state_dict)

def infer(cORs, ep, pp, input):
    '''
    DNN model inference
    :param cORs: client or server, 0 is client and 1 is server
    :param pp: partition point
    :param ep: exit point
    :return: intermediate data or final result
    '''
    netPair = 'NetExit' + str(ep) + 'Part' + str(pp)
    net = eval(netPair)[cORs]()

    # load params
    LOrR = 'L' if cORs == CLIENT else 'R'
    params_path = MODEL_DIR + netPair + LOrR + ".pth"
    load_model_param(net, params_path)

    net.eval()

    return net(input)


if __name__ == "__main__":
    input = torch.rand(1, 64, 48, 64).to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    out = infer(1, 1, 1, input)
    print("ok")
