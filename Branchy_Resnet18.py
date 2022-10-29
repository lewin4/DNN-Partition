# import os
import os.path
from Resnet_Model_Pair import *
import torch
import math
from torch import Tensor
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
# import torch.optim as optim
# import torch.nn.functional as F
from torch.utils import data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from collections import OrderedDict
# from torchvision.models import resnet18
from typing import Type, Any, Callable, Union, List, Optional
from FI_loader import get_loaders, SewageDataset
from collections import OrderedDict
from config import OUTPUT_DIR
from config import MODEL_DIR
import socket

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKER = 2
NUM_CLASSES = 6  # 10 classes for Cifar-10 dataset
learning_rate = 0.01
# branch = 2
# RESUME = OUTPUT_DIR + "checkpoint.pth"
RESUME = None
if socket.gethostname() == 'LAPTOP-5G1BF2CK':
    TRAIN_DATASET = r"D:\Code\data\sewage\classification_aug"
    TEST_DATASET = r"D:\Code\data\sewage\test_dataset"
elif socket.gethostname() == 'DESKTOP-D6L914M':
    TRAIN_DATASET = r"E:\LY\data\classification_aug"
    TEST_DATASET = r"E:\LY\data\test_dataset"

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        branch: int = 4,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block
        self.num_classes = num_classes

        #branch is 4 default
        self._branch = branch
        self.inplanes = 64

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = block(64, 64, 1, None, self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.block2 = block(64, 64, 1, None, self.groups, self.base_width,
                            self.dilation, self._norm_layer)

        # set branch

        # branch 1
        if self._branch >= 1:
            if self._branch == 1:
                self.branch1conv1 = conv1x1(64, 32)
                self.branch1bn1 = self._norm_layer(32)
                self.branch1fc = nn.Linear(24576, num_classes)

        # branch 2
        if self._branch >= 2:
            self.block3 = block(64, 128, 2,
                                nn.Sequential(
                                    conv1x1(64, 128, 2),
                                    norm_layer(128)),
                                self.groups, self.base_width,
                                self.dilation, self._norm_layer)
            self.block4 = block(128, 128, 1, None, self.groups, self.base_width,
                                self.dilation, self._norm_layer)
            if self._branch == 2:
                self.branch2conv1 = conv1x1(128, 32)
                self.branch2bn1 = self._norm_layer(32)
                self.branch2fc = nn.Linear(24576, num_classes)

        # branch 3
        if self._branch >= 3:
            self.block5 = block(128, 256, 2,
                                nn.Sequential(
                                    conv1x1(128, 256, 2),
                                    norm_layer(256)),
                                self.groups, self.base_width,
                                self.dilation, self._norm_layer)
            self.block6 = block(256, 256, 1, None, self.groups, self.base_width,
                                self.dilation, self._norm_layer)
            if self._branch == 3:
                self.branch3conv1 = conv1x1(256, 128)
                self.branch3bn1 = self._norm_layer(128)
                self.branch3fc = nn.Linear(24576, num_classes)

        # branch 4
        if self._branch >= 4:
            self.block7 = block(256, 512, 2,
                                nn.Sequential(
                                    conv1x1(256, 512, 2),
                                    norm_layer(512)),
                                self.groups, self.base_width,
                                self.dilation, self._norm_layer)
            self.block8 = block(512, 512, 1, None, self.groups, self.base_width,
                                self.dilation, self._norm_layer)
            if self._branch == 4:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.branch1fc = nn.Linear(64, num_classes)
        # self.block3 = block(64, 128, 2,
        #                     nn.Sequential(
        #                         conv1x1(64, 128, 2),
        #                         norm_layer(128)),
        #                     self.groups, self.base_width,
        #                     self.dilation, self._norm_layer)
        # self.block4 = block(128, 128, 1, None, self.groups, self.base_width,
        #                     self.dilation, self._norm_layer)
        # self.branch2fc = nn.Linear(128, num_classes)
        # self.block5 = block(128, 256, 2,
        #                     nn.Sequential(
        #                         conv1x1(128, 256, 2),
        #                         norm_layer(256)),
        #                     self.groups, self.base_width,
        #                     self.dilation, self._norm_layer)
        # self.block6 = block(256, 256, 1, None, self.groups, self.base_width,
        #                     self.dilation, self._norm_layer)
        # self.branch3fc = nn.Linear(256, num_classes)
        # self.block7 = block(256, 512, 2,
        #                     nn.Sequential(
        #                         conv1x1(256, 512, 2),
        #                         norm_layer(512)),
        #                     self.groups, self.base_width,
        #                     self.dilation, self._norm_layer)
        # self.block8 = block(512, 512, 1, None, self.groups, self.base_width,
        #                     self.dilation, self._norm_layer)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # branch 1
        x = self.block1(x)
        x = self.block2(x)
        if self._branch == 1:
            x1 = self.branch1conv1(x)
            x1 = self.branch1bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)
            x1 = x1.view(-1, 24576)
            x1 = self.branch1fc(x1)
            return x1

        # branch 2
        x = self.block3(x)
        x = self.block4(x)
        if self._branch == 2:
            x2 = self.branch2conv1(x)
            x2 = self.branch2bn1(x2)
            x2 = self.relu(x2)
            x2 = x2.view(-1, 24576)
            x2 = self.branch2fc(x2)
            return x2

        # branch 3
        x = self.block5(x)
        x = self.block6(x)
        if self._branch == 3:
            x3 = self.branch3conv1(x)
            x3 = self.branch3bn1(x3)
            x3 = self.relu(x3)
            x3 = x3.view(-1, 24576)
            x3 = self.branch3fc(x3)
            return x3

        # branch 4
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class LR_Scheduler(object):
    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def step_epoch(self) -> bool:
        return False

    def step_batch(self) -> bool:
        return False

    def step(self, metric: Optional[float] = None) -> None:
        if self.lr_scheduler is None:
            return

        if metric is None:
            self.lr_scheduler.step()
        else:
            self.lr_scheduler.step(metric)


class CosineAnnealingLR(LR_Scheduler):
    def __init__(
            self, optimizer: torch.optim.Optimizer,
            n_epochs: int,
            n_batches: int,
            eta_min: float,
            last_epoch: int = -1
    ):
        t_max = n_batches * n_epochs
        base_lr = optimizer.param_groups[0]["lr"]
        last_epoch = (last_epoch + 1) * n_batches - 1

        # This is to bypass pytorch's weird requirements for last_epoch.
        # This is the closed form directly taken from
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        if last_epoch > 0:
            learning_rate = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * last_epoch / t_max)) / 2
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        super().__init__(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min, last_epoch=last_epoch))

    def step_batch(self) -> bool:
        return True


if __name__ == "__main__":
    for branch in range(1, 5):
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=6, branch=branch).to(device)
        print("Resnet18 is created.")
        print(model)
        train_loader, val_loader = get_loaders(TRAIN_DATASET,
                                             BATCH_SIZE,
                                             [192, 256],
                                             num_workers=NUM_WORKER)

        test_dataset = SewageDataset(TEST_DATASET, mode="test")
        test_loader = data.DataLoader(test_dataset,
                                      BATCH_SIZE,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=NUM_WORKER)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = learning_rate
        print('Optimizer created')

        criterion = torch.nn.CrossEntropyLoss()
        n_batch_size = len(train_loader)
        last_epoch = 0
        if RESUME is not None:
            checkpoint = torch.load(RESUME, map_location=device)
            last_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        lr_scheduler = CosineAnnealingLR(
            optimizer, NUM_EPOCHS, n_batch_size, eta_min=1.e-6, last_epoch=last_epoch)
        print('LR Scheduler created')

        # start training!!
        print('Starting training...')
        model.train()
        total_steps = 1
        end = False
        time_local = time.localtime()
        time_str = str(time_local[1]) + "m" + str(time_local[2]) + "d" + str(time_local[3]) + "h" + str(
            time_local[4]) + "m" + str(time_local[5]) + "s"
        writer_dir = OUTPUT_DIR + "logs/branch" + str(branch) + "/" + time_str + "/"
        summary_writer = SummaryWriter(writer_dir)
        best_acc = 0.
        from tqdm import tqdm
        for epoch in range(last_epoch, NUM_EPOCHS):
            for imgs, classes in tqdm(train_loader, desc="Train epoch: {}".format(epoch)):
                imgs, classes = imgs.to(device), classes.to(device)
                optimizer.zero_grad()
                # calculate the loss
                output = model(imgs)
                loss = criterion(output, classes)
                summary_writer.add_scalar("train loss", loss.item(), total_steps)
                # update the parameters
                loss.backward()
                optimizer.step()

                if total_steps % 50 == 0:
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                          .format(epoch + 1, total_steps, loss.item(), accuracy.item()))

                if total_steps % 100 == 0:

                    # ~~~~~~~VALIDATION~~~~~~~~~
                    print("Val {}".format(total_steps))
                    correct_count = 0
                    total_count = 0
                    model.eval()
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        with torch.no_grad():  # no gradient descent!
                            logps = model(images)

                        logps = logps.detach()
                        targets = labels.detach()

                        _, predicted = logps.max(dim=1)
                        total_count += torch.tensor(targets.size(0), dtype=torch.float)
                        correct_count += predicted.eq(targets).cpu().float().sum()

                    print("Number Of Images Tested =", total_count)
                    acc = (correct_count / total_count)
                    if acc >= best_acc:
                        best_acc = acc
                        if not os.path.exists(MODEL_DIR):
                            os.makedirs(MODEL_DIR)
                        state_dict = model.state_dict()
                        torch.save(state_dict, MODEL_DIR + 'branch' + str(branch) + '_best_model.pth')
                        print("Saved best acc model: {}".format(best_acc))
                    print("Model Accuracy =", acc)
                    summary_writer.add_scalar("val acc", acc, total_steps)
                    if acc > 0.95:
                        end = True
                    model.train()
                if end:
                    break

                total_steps += 1
            if end or (epoch == (NUM_EPOCHS-1)):
                state_dict = torch.load(MODEL_DIR + 'branch' + str(branch) + '_best_model.pth', map_location=device)
                from Time_Prediction import partition_point_number
                for partition_point in range(partition_point_number[branch-1]):
                    L_model_name = "NetExit" + str(branch) + "Part" + str(partition_point + 1) + 'L'
                    R_model_name = "NetExit" + str(branch) + "Part" + str(partition_point + 1) + 'R'
                    net_L = eval(L_model_name)()
                    net_L_state_dict = net_L.state_dict()
                    assert set(net_L_state_dict.keys()).issubset(set(state_dict.keys()))
                    net_l_state_dict = OrderedDict({key: state_dict[key] for key in net_L_state_dict.keys()})
                    torch.save(net_l_state_dict, MODEL_DIR + L_model_name + ".pth")
                    net_R = eval(R_model_name)()
                    net_R_state_dict = net_R.state_dict()
                    assert set(net_R_state_dict.keys()).issubset(set(state_dict.keys()))
                    net_r_state_dict = OrderedDict({key: state_dict[key] for key in net_R_state_dict.keys()})
                    torch.save(net_r_state_dict, MODEL_DIR + R_model_name + ".pth")
                break
            lr_scheduler.step()

            if epoch % 1 == 0:
                if not os.path.exists(OUTPUT_DIR):
                    os.makedirs(OUTPUT_DIR)
                torch.save({"state_dict": model.state_dict(),
                            "epoch": epoch,
                            "optimizer": optimizer.state_dict()}, OUTPUT_DIR + 'checkpoint.pth')
