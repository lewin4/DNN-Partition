from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear, MaxPool2d
import torch
from collections import OrderedDict
from typing import Tuple
import numpy as np
from regression import get_regression_model

from config import *

# partitiion point number for every branch
partition_point_number = [2, 2, 3, 4]


def get_layer_data(net: torch.nn.Module,
                   input_size,
                   device: str = "cuda") -> (OrderedDict, Tuple):
    def register_hook(name, module):
        def hook(module: torch.nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):

            if isinstance(module, Conv2d):
                x1 = module.in_channels
                x2 = (module.kernel_size[0] * module.kernel_size[1] / module.stride[0]) ** 2 * module.out_channels
                layer_data[Conv2d].append([name, x1, x2])
            elif isinstance(module, ReLU) or isinstance(module, BatchNorm2d):
                input_shape = input[0].size()
                x1 = np.prod(input_shape)
                layer_data[type(module)].append([name, x1])
            elif isinstance(module, MaxPool2d) or isinstance(module, Linear):
                input_shape = input[0].size()
                output_shape = output.size()
                x1 = np.prod(input_shape)
                x2 = np.prod(output_shape)
                layer_data[type(module)].append([name, x1, x2])

        if (not isinstance(module, torch.nn.Sequential)
                and not isinstance(module, torch.nn.ModuleList)
                and not (module == net)):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    layer_data = OrderedDict()
    layer_data[Conv2d] = []
    layer_data[ReLU] = []
    layer_data[BatchNorm2d] = []
    layer_data[MaxPool2d] = []
    layer_data[Linear] = []

    # register hook
    for name, module in net.named_modules():
        register_hook(name, module)

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    x = torch.rand(input_size).type(dtype)

    # make a forward pass
    # print(x.shape)
    y = net(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return layer_data, y.size()


###############################################
# Mobile device side time prediction class
###############################################
class DeviceTime:
    def __init__(self, client_regression_data):
        # regression data dict should have 6 type(conv, fc, pool, bn, load, relu)
        assert len(client_regression_data) == 6
        self.client_regression_data = client_regression_data
        self.conv = get_regression_model("conv")
        self.conv.load_state_dict(self.client_regression_data["conv"]["weight"])
        self.bn = get_regression_model("bn")
        self.bn.load_state_dict(self.client_regression_data["bn"]["weight"])
        self.pool = get_regression_model("pool")
        self.pool.load_state_dict(self.client_regression_data["pool"]["weight"])
        self.relu = get_regression_model("relu")
        self.relu.load_state_dict(self.client_regression_data["relu"]["weight"])
        self.load = get_regression_model("load")
        self.load.load_state_dict(self.client_regression_data["load"]["weight"])
        self.linear = get_regression_model("fc")
        self.linear.load_state_dict(self.client_regression_data["fc"]["weight"])
        # for layer_type, data in client_regression_data.items():
        #     if layer_type in ["conv", "fc", "pool"]:
        #         x1_min, x2_min = data["x_min"][0], data["x_min"][1]
        #         x1_dif = data["x_max"][0] - x1_min
        #         x2_dif = data["x_max"][1] - x2_min
        #         y_min = data["y_min"][0]
        #         y_dif = data["y_max"][0] - y_min
        #         w1, w2 = data["weight"][0], data["weight"][1]
        #         W1 = w1/x1_dif*y_dif
        #         W2 = w2/x2_dif*y_dif
        #         B = (data["weight"][2]) * y_dif + y_min - y_dif * (w1*x1_min/x1_dif + w2*x2_min/x2_dif)
        #         if layer_type == "conv":
        #             self.conv_w1 = W1
        #             self.conv_w2 = W2
        #             self.conv_b = B
        #         elif layer_type == "fc":
        #             self.fc_w1 = W1
        #             self.fc_w2 = W2
        #             self.fc_b = B
        #         elif layer_type == "pool":
        #             self.pool_w1 = W1
        #             self.pool_w2 = W2
        #             self.pool_b = B
        #     else:
        #         x1_min= data["x_min"][0]
        #         x1_dif = data["x_max"][0] - x1_min
        #         y_min = data["y_min"][0]
        #         y_dif = data["y_max"][0] - y_min
        #         w1= data["weight"][0]
        #         W1 = w1 / x1_dif * y_dif
        #         B = (data["weight"][1]) * y_dif + y_min - (y_dif * w1 * x1_min / x1_dif)
        #         if layer_type == "bn":
        #             self.bn_w1 = W1
        #             self.bn_b = B
        #         elif layer_type == "load":
        #             self.load_w1 = W1
        #             self.load_b = B
        #         elif layer_type == "relu":
        #             self.relu_w1 = W1
        #             self.relu_b = B

    def data_to_one(self, layer_type: str, data):
        if layer_type in ["conv", "fc", "pool"]:
            assert len(data) == 2
            data_out = list()
            x1_max = self.client_regression_data[layer_type]["x_max"][0]
            x1_min = self.client_regression_data[layer_type]["x_min"][0]
            x2_max = self.client_regression_data[layer_type]["x_max"][1]
            x2_min = self.client_regression_data[layer_type]["x_min"][1]
            data_out.append((data[0] - x1_min) / (x1_max - x1_min))
            data_out.append((data[1] - x2_min) / (x2_max - x2_min))
        elif layer_type in ["relu", "load", "bn"]:
            assert len(data) == 1
            data_out = list()
            x1_max = self.client_regression_data[layer_type]["x_max"][0]
            x1_min = self.client_regression_data[layer_type]["x_min"][0]
            data_out.append((data[0] - x1_min) / (x1_max - x1_min))
        else:
            raise Exception("类型错误{}".format(layer_type))
        return data_out

    def one_to_data(self, layer_type: str, data):
        y_max = self.client_regression_data[layer_type]["y_max"][0]
        y_min = self.client_regression_data[layer_type]["y_min"][0]
        data_out = data[0] * (y_max - y_min) + y_min
        return data_out

    # time predict function
    def device_bn(self, data_size):
        data_size = self.data_to_one("bn", [data_size])
        data_out = self.bn(torch.DoubleTensor(data_size).resize(1, 1)).item()
        return self.one_to_data("bn", [data_out])
        # return self.bn_w1 * data_size + self.bn_b

    def device_pool(self, input_data_size, output_data_size):
        data = self.data_to_one("pool", [input_data_size, output_data_size])
        data_out = self.pool(torch.DoubleTensor(data).resize(1, 2)).item()
        return  self.one_to_data("pool", [data_out])
        # return self.pool_w1 * input_data_size + self.pool_w2 * output_data_size + self.pool_b

    def device_relu(self, input_data_size):
        data = self.data_to_one("relu", [input_data_size])
        data_out = self.relu(torch.DoubleTensor(data).resize(1, 1)).item()
        return self.one_to_data("relu", [data_out])
        # return self.relu_w1 * input_data_size + self.relu_b

    def device_dropout(self, input_data_size):
        return 9.341929545685408e-08 * input_data_size + 0.0007706006740869353

    def device_linear(self, input_data_size, output_data_size):
        data = self.data_to_one("fc", [input_data_size, output_data_size])
        data_out = self.linear(torch.DoubleTensor(data).resize(1, 2)).item()
        return self.one_to_data("fc", [data_out])
        # return self.fc_w1 * input_data_size + self.fc_w2 * output_data_size + self.fc_b

    def device_conv(self, feature_map_amount, compution_each_pixel):
        # compution_each_pixel stands for (filter size / stride)^2 * (number of filters)
        data = self.data_to_one("conv", [feature_map_amount, compution_each_pixel])
        data = torch.DoubleTensor(data).resize(1, 2)
        data_out = self.conv(data)
        data_out = data_out.item()
        return self.one_to_data("conv", [data_out])
        # return self.conv_w1 * feature_map_amount + self.conv_w2 * compution_each_pixel + self.conv_b

    def device_model_load(self, model_size):
        data = self.data_to_one("load", [model_size])
        data_out = self.load(torch.DoubleTensor(data).resize(1, 1)).item()
        return self.one_to_data("load", [data_out])
        # return self.load_w1 * model_size + self.load_b

    # tool
    def predict_time(self, net: torch.nn.Module,
                     input_size,
                     device: str = "cuda"):
        '''
        :param branch_number: the index of branch
        :param partition_point_number: the index of partition point
        :return:
        '''
        layer_data, output_size = get_layer_data(net, input_size, device)

        time = 0
        for layer_type, items in layer_data.items():
            if layer_type == Conv2d:
                for item in items:
                    dtime = self.device_conv(*item[1:])
                    time += (dtime if dtime > 0 else 0)
            if layer_type == MaxPool2d:
                for item in items:
                    dtime = self.device_pool(*item[1:])
                    time += (dtime if dtime > 0 else 0)
            if layer_type == Linear:
                for item in items:
                    dtime = self.device_linear(*item[1:])
                    time += (dtime if dtime > 0 else 0)
            if layer_type == BatchNorm2d:
                for item in items:
                    dtime = self.device_bn(*item[1:])
                    time += (dtime if dtime > 0 else 0)
            if layer_type == ReLU:
                for item in items:
                    dtime = self.device_relu(*item[1:])
                    time += (dtime if dtime > 0 else 0)
        return time, output_size


###############################################
# Edge server side time prediction class
###############################################
class ServerTime(DeviceTime):
    def __init__(self, server_regression_data):
        super(ServerTime, self).__init__(server_regression_data)

    # tool
    def predict_time(self, net: torch.nn.Module,
                     input_size,
                     device: str = "cuda"):
        """
        写一个钩子函数，在钩子函数中根据输入大小，输出大小，以及假如是卷积层那就根据卷积的层的规格
        直接调用实例函数把每一层的层类型，输入大小，输出大小，计算时间都输出到列表里
        List[Dict[类型：List[输入大小, 输出大小, 计算时间]],]
        仿照torchsummary写，勾出来之后
        """
        layer_data, _ = get_layer_data(net, input_size, device)

        time = 0
        convtime = 0
        pooltime = 0
        lineartime = 0
        bntime = 0
        relutime = 0
        for layer_type, items in layer_data.items():
            if layer_type == Conv2d:
                for item in items:
                    dtime = self.device_conv(*item[1:])
                    convtime += (dtime if dtime > 0 else 0)
            if layer_type == MaxPool2d:
                for item in items:
                    dtime = self.device_pool(*item[1:])
                    pooltime += (dtime if dtime > 0 else 0)
            if layer_type == Linear:
                for item in items:
                    dtime = self.device_linear(*item[1:])
                    lineartime += (dtime if dtime > 0 else 0)
            if layer_type == BatchNorm2d:
                for item in items:
                    dtime = self.device_bn(*item[1:])
                    bntime += (dtime if dtime > 0 else 0)
            if layer_type == ReLU:
                for item in items:
                    dtime = self.device_relu(*item[1:])
                    relutime += (dtime if dtime > 0 else 0)
        time = convtime + pooltime + lineartime + bntime + relutime
        return time

    def server_model_load(self, right_model_size):
        return self.device_model_load(right_model_size)


class OutputSizeofPartitionLayer:
    # float32 which is 4B(32 bits)
    branch1 = {
        'pool0': 64 * 15 * 15 * 32,
        'pool1': 32 * 7 * 7 * 32,
    }
    branch2 = {
        'pool0': 64 * 15 * 15 * 32,
        'pool1': 192 * 6 * 6 * 32,
        'pool2': 32 * 2 * 2 * 32,
    }
    branch3 = {
        'pool0': 64 * 15 * 15 * 32,
        'pool1': 192 * 6 * 6 * 32,
        'pool2': 256 * 2 * 2 * 32,
    }
    branches = [branch1, branch2, branch3]

    @classmethod
    def output_size(cls, branch_number, partition_point_number):
        '''
        :return:unit(bit)
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point = partition_point_index_set[partition_point_number]
        # layers in partitioned model
        layer = branch_layer[partition_point]
        outputsize_dict = cls.branches[branch_number]
        return outputsize_dict[layer]
