import os.path

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from tensorboardX import SummaryWriter
from Branchy_Resnet18 import CosineAnnealingLR
from typing import Dict
from Resnet_Model_Pair import *
from torchsummary import summary
from config import regression_type, REGRESSION_RESULT_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2.使用Class设计模型
class LinearModel(torch.nn.Module):

    def __init__(self, in_channel, out_channel):
        super(LinearModel, self).__init__()
        self.liner = torch.nn.Linear(in_channel, out_channel, dtype=torch.double)  # (1,1)表示输入和输出的维度

    def forward(self, x):
        y_pred = self.liner(x)
        return y_pred


def get_regression_model(type: str) -> torch.nn.Module:
    if type in ["conv", "pool", "fc"]:
        return LinearModel(2, 1)
    else:
        return LinearModel(1, 1)


def get_data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


def get_a_randint(a, b, seed=None):
    # 得到一个在a 和b 之间的整数，如果seed 被指定，下一个被得到的数是固定的
    if seed is not None:
        random.seed(seed)
    return random.randint(a, b)


def data_to_one(data_x, data_y) -> Dict:
    min_data_x, _ = torch.min(data_x, dim=0, keepdim=True)
    max_data_x, _ = torch.max(data_x, dim=0, keepdim=True)
    min_data_y, _ = torch.min(data_y, dim=0, keepdim=True)
    max_data_y, _ = torch.max(data_y, dim=0, keepdim=True)
    data_x = (data_x - min_data_x) / (max_data_x - min_data_x)
    data_y = (data_y - min_data_y) / (max_data_y - min_data_y)

    data = {
        "x_max": max_data_x,
        "x_min": min_data_x,
        "y_max": max_data_y,
        "y_min": min_data_y,
        "data_x": data_x,
        "data_y": data_y,
    }
    return data


def generate_conv_data_and_save(path: str) -> Dict:
    # 生成回归卷积层所需要的数据，并把数据保存在path中
    # x形状(n, 2), y形状(n, 1) 且该xy经过了maxmin归一化
    print("Generating conv data ....")
    conv3s164 = torch.nn.Conv2d(64, 64, 3, 1)
    conv3s1128 = torch.nn.Conv2d(128, 128, 3, 1)
    conv3s1256 = torch.nn.Conv2d(256, 256, 3, 1)
    conv3s264 = torch.nn.Conv2d(64, 128, 3, 2)
    conv3s2256 = torch.nn.Conv2d(256, 512, 3, 2)
    conv164 = torch.nn.Conv2d(64, 128, 1)
    conv1128 = torch.nn.Conv2d(128, 256, 1)
    conv1256 = torch.nn.Conv2d(256, 512, 1)

    conv_list = {"conv3s164": [conv3s164, 64, 9 ** 2 * 64],
                 "conv3s1128": [conv3s1128, 128, 9 ** 2 * 128],
                 "conv3s1256": [conv3s1256, 256, 9 ** 2 * 256],
                 "conv3s264": [conv3s264, 64, float(9 / 2) ** 2 * 128],
                 "conv3s2256": [conv3s2256, 256, float(9 / 2) ** 2 * 512],
                 "conv164": [conv164, 64, 128],
                 "conv1128": [conv1128, 128, 256],
                 "conv1256": [conv1256, 256, 512]
                 }

    data_x = torch.DoubleTensor(0, 2)
    data_y = torch.DoubleTensor(0, 1)
    for i in range(50):
        for key, value in conv_list.items():
            print(i, key)
            conv = value[0].eval().to(device)
            x1 = value[1]
            x2 = value[2]
            data = torch.rand(8, conv.in_channels, 192, 256).to(device)
            with torch.no_grad():
                time_start = time.time()
                y = conv(data)
                times = time.time() - time_start
            data_x = torch.cat((data_x, torch.tensor([[x1, x2]], dtype=torch.float32)), 0)
            data_y = torch.cat((data_y, torch.tensor([[times]])), 0)
    print("Data has been generated.")

    print("Data to 0 - 1.")
    conv_train_data = data_to_one(data_x, data_y)

    torch.save(conv_train_data, path)
    return conv_train_data


def generate_relu_data_and_save(path):
    print("Generating relu data ... ")
    relu = torch.nn.ReLU().eval().to(device)

    data_x = torch.DoubleTensor(0, 1)
    data_y = torch.DoubleTensor(0, 1)
    for batch in range(400):
        b = random.randint(5, 16)
        c = random.randint(3, 512)
        h = random.randint(8, 128)
        w = h
        print("b, c, h, w: ", b, c, h, w)
        batch_tensor = torch.rand((b, c, h, w)).to(device)
        x = np.prod(batch_tensor.size())
        print("x: ", x)
        with torch.no_grad():
            time_start = time.time()
            y = relu(batch_tensor)
            times = time.time() - time_start
        data_y = torch.cat((data_y, torch.tensor([[times]])), 0)
        data_x = torch.cat((data_x, torch.tensor([[x]])), 0)

    print("Data has been generated.")

    print("Data to 0 - 1.")
    relu_train_data = data_to_one(data_x, data_y)

    torch.save(relu_train_data, path)
    return relu_train_data


def generate_bn_data_and_save(path):
    print("Generating BN data ... ")
    bn_layers = [
        torch.nn.BatchNorm2d(3),
        torch.nn.BatchNorm2d(64),
        torch.nn.BatchNorm2d(128),
        torch.nn.BatchNorm2d(256),
        torch.nn.BatchNorm2d(512),
    ]
    bn_channel = [3, 64, 128, 256, 512]

    data_x = torch.DoubleTensor(0, 1)
    data_y = torch.DoubleTensor(0, 1)
    for batch in range(800):
        b = random.randint(5, 16)
        c_id = random.randint(0, 4)
        c = bn_channel[c_id]
        h = random.randint(8, 128)
        w = h
        batch_tensor = torch.rand((b, c, h, w)).to(device)
        print("b, c, h, w: ", b, c, h, w)
        x = np.prod(batch_tensor.size())
        with torch.no_grad():
            time_start = time.time()
            y = bn_layers[c_id].to(device).eval()(batch_tensor)
            times = time.time() - time_start
        data_y = torch.cat((data_y, torch.tensor([[times]])), 0)
        data_x = torch.cat((data_x, torch.tensor([[x]])), 0)

    print("Data has been generated.")

    print("Data to 0 - 1.")
    bn_train_data = data_to_one(data_x, data_y)

    torch.save(bn_train_data, path)
    return bn_train_data


def generate_pool_data_and_save(path):
    print("Generating pool data ... ")
    pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device).eval()

    data_x = torch.DoubleTensor(0, 2)
    data_y = torch.DoubleTensor(0, 1)

    for batch in range(400):
        b = random.randint(5, 16)
        c = random.randint(3, 512)
        h = random.randint(8, 128)
        w = h
        print("b, c, h, w: ", b, c, h, w)
        batch_tensor = torch.rand((b, c, h, w)).to(device)
        x0 = np.prod(batch_tensor.size())
        with torch.no_grad():
            time_start = time.time()
            y = pool(batch_tensor)
            times = time.time() - time_start
        x1 = np.prod(y.size())
        data_y = torch.cat((data_y, torch.tensor([[times]])), 0)
        data_x = torch.cat((data_x, torch.tensor([[x0, x1]])), 0)

    print("Data has been generated.")

    print("Data to 0 - 1.")
    pool_train_data = data_to_one(data_x, data_y)

    torch.save(pool_train_data, path)
    return pool_train_data


def generate_fc_data_and_save(path):
    print("Generating fc data ... ")

    data_x = torch.DoubleTensor(0, 2)
    data_y = torch.DoubleTensor(0, 1)

    for batch in range(400):
        h = random.randint(128, 24576)
        w = random.randint(8, 128)
        print("h, w: ", h, w)
        fc = torch.nn.Linear(h, w).to(device).eval()
        batch_tensor = torch.rand((16, h)).to(device)
        x0 = np.prod(batch_tensor.size())
        with torch.no_grad():
            time_start = time.time()
            y = fc(batch_tensor)
            times = time.time() - time_start
        x1 = np.prod(y.size())
        data_y = torch.cat((data_y, torch.tensor([[times]])), 0)
        data_x = torch.cat((data_x, torch.tensor([[x0, x1]])), 0)

    print("Data has been generated.")

    print("Data to 0 - 1.")
    fc_train_data = data_to_one(data_x, data_y)

    torch.save(fc_train_data, path)
    return fc_train_data


def generate_load_data_and_save(path):
    print("Generating load data ... ")

    data_x = torch.DoubleTensor(0, 1)
    data_y = torch.DoubleTensor(0, 1)
    from Time_Prediction import partition_point_number
    for i in range(19):
        for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):  # iter[2, 1, 0]
            for partition_point in range(partition_point_number[exit_branch]):
                L_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'L'
                R_model_name = "NetExit" + str(exit_branch + 1) + "Part" + str(partition_point + 1) + 'R'

                net_L = eval(L_model_name)().eval().to(device)
                summarydict, summ = summary(net_L, INPUT_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")
                left_model_size = summarydict["Total params"]
                data_x = torch.cat((data_x, torch.tensor([[left_model_size]])), 0)
                time_start = time.time()
                net_L.load_state_dict(torch.load(MODEL_DIR + L_model_name + ".pth", map_location=device))
                times = time.time() - time_start
                data_y = torch.cat((data_y, torch.tensor([[times]])), 0)
                del net_L

                net_R = eval(R_model_name)().eval().to(device)
                output_shape = next(reversed(summ.items()))[1]["output_shape"]
                img_shape = tuple(output_shape[1:])
                summarydict, _ = summary(net_R, img_shape, device="cuda" if torch.cuda.is_available() else "cpu")
                right_model_size = summarydict["Total params"]
                data_x = torch.cat((data_x, torch.tensor([[right_model_size]])), 0)
                time_start = time.time()
                net_R.load_state_dict(torch.load(MODEL_DIR + R_model_name + ".pth", map_location=device))
                times = time.time() - time_start
                data_y = torch.cat((data_y, torch.tensor([[times]])), 0)

    print("Data has been generated.")

    print("Data to 0 - 1.")
    load_train_data = data_to_one(data_x, data_y)

    torch.save(load_train_data, path)
    return load_train_data


def load_train_data(path):
    checkpoint = torch.load(path)
    return checkpoint


def save_regression_result(model: torch.nn.Module,
                           type: str,
                           max_data_x, min_data_x,
                           max_data_y, min_data_y,
                           path_dir: str):
    with open(path_dir + type + ".txt", "w", encoding='utf-8') as f:
        f.write("Weight and bias:")
        for param in model.parameters():
            f.write(str(param.cpu().detach().numpy()) + " ")
        f.write("\nx max and min: {}, {}\n".format(max_data_x, min_data_x))
        f.write("y max and min: {}, {}\n".format(max_data_y, min_data_y))
    print("Weight and Bias: \n", model.liner.weight, model.liner.bias)

    checkpoint = {
        "weight": model.state_dict(),
        "x_max": max_data_x,
        "x_min": min_data_x,
        "y_max": max_data_y,
        "y_min": min_data_y,
    }
    torch.save(checkpoint, path_dir + "/" + type + ".pth")
    print("model is saved")


def regression(type: str, num_epochs: int = 15):
    train_data_dir = "./logs/train_data/"
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    train_data_path = train_data_dir + type + "_train_data.pth"
    if os.path.exists(train_data_path):
        checkpoint = load_train_data(train_data_path)
    else:
        checkpoint = eval("generate_" + type + "_data_and_save")(train_data_path)
    data_x = checkpoint["data_x"]
    data_y = checkpoint["data_y"]
    max_data_x = checkpoint["x_max"]
    max_data_y = checkpoint["y_max"]
    min_data_x = checkpoint["x_min"]
    min_data_y = checkpoint["y_min"]

    model = get_regression_model(type).to(device)  # 创建类LinearModel的实例
    print("Init weight and bias: \n", model.liner.weight, "\n", model.liner.bias)
    stat = model.state_dict()
    # 3.构建损失函数和优化器的选择
    print("Starting "+ type + " regression...")
    batch_size = 10

    n_batch_size = int(data_x.size()[0] / batch_size)
    time_local = time.localtime()
    time_str = str(time_local[1]) + "m" + str(time_local[2]) + "d" + str(time_local[3]) + "h" + str(
        time_local[4]) + "m" + str(time_local[5]) + "s"
    writer_dir = "./logs/generate_data/" + type + "/" + time_str + "/"
    summary_writer = SummaryWriter(writer_dir)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    lr_scheduler = CosineAnnealingLR(optimizer, num_epochs, n_batch_size, eta_min=1.e-6, last_epoch=-1)

    n_iter = 0
    model.train()
    for epoch in range(num_epochs):
        for x, y in get_data_iter(batch_size, data_x, data_y):
            y_pred = model(x.to(device))
            y = y.to(device)
            loss = criterion(y_pred, y)
            summary_writer.add_scalar("regression loss", loss.item(), n_iter)
            n_iter = n_iter + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 进行更新update
        lr_scheduler.step()

    print("regression OK!")

    if not os.path.exists(REGRESSION_RESULT_DIR):
        os.makedirs(REGRESSION_RESULT_DIR)
    save_regression_result(model,
                           type,
                           max_data_x, min_data_x,
                           max_data_y, min_data_y,
                           REGRESSION_RESULT_DIR)


if __name__ == "__main__":
    print("="*20)
    print("This is regression stage.")
    print("="*20)
    # for layer_type in regression_type:
    regression("conv")
