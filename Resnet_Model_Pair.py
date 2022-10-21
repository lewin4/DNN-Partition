import torch.nn as nn
from torch import Tensor
import torch
from config import *
from collections import OrderedDict
from typing import Optional, Callable


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


####################
# Exit 1 Part 1
####################
class NetExit1Part1L(nn.Module):
    '''
    Resnet Exit at branch 1 and part at point 1, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class NetExit1Part1R(nn.Module):
    '''
    Resnet Exit at branch 1 and part at point 1, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.block1 = BasicBlock(64, 64, 1, None, 1, 64,
                                 1, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, 1, 64,
                                 1, self._norm_layer)

        self.branch1conv1 = conv1x1(64, 32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch1bn1 = self._norm_layer(32)
        self.branch1fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x1 = self.branch1conv1(x)
        x1 = self.branch1bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = x1.view(-1, 24576)
        x1 = self.branch1fc(x1)
        return x1


####################
# Exit 1 Part 2
####################
class NetExit1Part2L(nn.Module):
    '''
    Resnet Exit at branch 1 and part at point 2, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, 1, 64,
                                 1, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        return x


class NetExit1Part2R(nn.Module):
    '''
    Resnet Exit at branch 1 and part at point 2, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.block2 = BasicBlock(64, 64, 1, None, 1, 64,
                                 1, self._norm_layer)

        self.branch1conv1 = conv1x1(64, 32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch1bn1 = self._norm_layer(32)
        self.branch1fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block2(x)
        x1 = self.branch1conv1(x)
        x1 = self.branch1bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = x1.view(-1, 24576)
        x1 = self.branch1fc(x1)
        return x1


####################
# Exit 2 Part 1
####################
class NetExit2Part1L(nn.Module):
    '''
    Reset Exit at branch 2 and part at point 1, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class NetExit2Part1R(nn.Module):
    '''
    Resnet Exit at branch 2 and part at point 1, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block3 = BasicBlock(64, 128, 2,
                            nn.Sequential(
                                conv1x1(64, 128, 2),
                                nn.BatchNorm2d(128)),
                            self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.branch2conv1 = conv1x1(128, 32)
        self.branch2bn1 = self._norm_layer(32)
        self.branch2fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x2 = self.branch2conv1(x)
        x2 = self.branch2bn1(x2)
        x2 = self.relu(x2)
        x2 = x2.view(-1, 24576)
        x2 = self.branch2fc(x2)
        return x2


####################
# Exit 2 Part 2
####################
class NetExit2Part2L(nn.Module):
    '''
    Resnet Exit at branch 2 and part at point 2, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class NetExit2Part2R(nn.Module):
    '''
    Resnet Exit at branch 2 and part at point 2, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.branch2conv1 = conv1x1(128, 32)
        self.branch2bn1 = self._norm_layer(32)
        self.branch2fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block3(x)
        x = self.block4(x)
        x2 = self.branch2conv1(x)
        x2 = self.branch2bn1(x2)
        x2 = self.relu(x2)
        x2 = x2.view(-1, 24576)
        x2 = self.branch2fc(x2)
        return x2


####################
# Exit 3 Part 1
####################
class NetExit3Part1L(nn.Module):
    '''
    Resnet Exit at branch 3 and part at point 1, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class NetExit3Part1R(nn.Module):
    '''
    Resnet Exit at branch 3 and part at point 1, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block5 = BasicBlock(128, 256, 2,
                            nn.Sequential(
                                conv1x1(128, 256, 2),
                                self._norm_layer(256)),
                            self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.branch3conv1 = conv1x1(256, 128)
        self.branch3bn1 = self._norm_layer(128)
        self.branch3fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x3 = self.branch3conv1(x)
        x3 = self.branch3bn1(x3)
        x3 = self.relu(x3)
        x3 = x3.view(-1, 24576)
        x3 = self.branch3fc(x3)
        return x3


####################
# Exit 3 Part 2
####################
class NetExit3Part2L(nn.Module):
    '''
    Resnet Exit at branch 3 and part at point 2, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class NetExit3Part2R(nn.Module):
    '''
    Resnet Exit at branch 3 and part at point 2, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block5 = BasicBlock(128, 256, 2,
                            nn.Sequential(
                                conv1x1(128, 256, 2),
                                self._norm_layer(256)),
                            self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.branch3conv1 = conv1x1(256, 128)
        self.branch3bn1 = self._norm_layer(128)
        self.branch3fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x3 = self.branch3conv1(x)
        x3 = self.branch3bn1(x3)
        x3 = self.relu(x3)
        x3 = x3.view(-1, 24576)
        x3 = self.branch3fc(x3)
        return x3


####################
# Exit 3 Part 3
####################
class NetExit3Part3L(nn.Module):
    '''
    Resnet Exit at branch 3 and part at point 3, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class NetExit3Part3R(nn.Module):
    '''
    Resnet Exit at branch 3 and part at point 3, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block5 = BasicBlock(128, 256, 2,
                                 nn.Sequential(
                                     conv1x1(128, 256, 2),
                                     self._norm_layer(256)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.branch3conv1 = conv1x1(256, 128)
        self.branch3bn1 = self._norm_layer(128)
        self.branch3fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = self.block5(x)
        x = self.block6(x)
        x3 = self.branch3conv1(x)
        x3 = self.branch3bn1(x3)
        x3 = self.relu(x3)
        x3 = x3.view(-1, 24576)
        x3 = self.branch3fc(x3)
        return x3


####################
# Exit 4 Part 1
####################
class NetExit4Part1L(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 1, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class NetExit4Part1R(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 1, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block5 = BasicBlock(128, 256, 2,
                                 nn.Sequential(
                                     conv1x1(128, 256, 2),
                                     self._norm_layer(256)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block7 = BasicBlock(256, 512, 2,
                            nn.Sequential(
                                conv1x1(256, 512, 2),
                                nn.BatchNorm2d(512)),
                            self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.block8 = BasicBlock(512, 512, 1, None, self.groups, self.base_width,
                            self.dilation, self._norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


####################
# Exit 4 Part 2
####################
class NetExit4Part2L(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 2, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class NetExit4Part2R(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 2, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block5 = BasicBlock(128, 256, 2,
                                 nn.Sequential(
                                     conv1x1(128, 256, 2),
                                     self._norm_layer(256)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block7 = BasicBlock(256, 512, 2,
                                 nn.Sequential(
                                     conv1x1(256, 512, 2),
                                     nn.BatchNorm2d(512)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block8 = BasicBlock(512, 512, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def forward(self, x):
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


####################
# Exit 4 Part 3
####################
class NetExit4Part3L(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 3, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class NetExit4Part3R(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 3, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block5 = BasicBlock(128, 256, 2,
                                 nn.Sequential(
                                     conv1x1(128, 256, 2),
                                     self._norm_layer(256)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block7 = BasicBlock(256, 512, 2,
                                 nn.Sequential(
                                     conv1x1(256, 512, 2),
                                     nn.BatchNorm2d(512)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block8 = BasicBlock(512, 512, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def forward(self, x):
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


####################
# Exit 4 Part 4
####################
class NetExit4Part4L(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 4, left part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block2 = BasicBlock(64, 64, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block3 = BasicBlock(64, 128, 2,
                                 nn.Sequential(
                                     conv1x1(64, 128, 2),
                                     nn.BatchNorm2d(128)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block4 = BasicBlock(128, 128, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block5 = BasicBlock(128, 256, 2,
                                 nn.Sequential(
                                     conv1x1(128, 256, 2),
                                     self._norm_layer(256)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block6 = BasicBlock(256, 256, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


class NetExit4Part4R(nn.Module):
    '''
    Resnet Exit at branch 4 and part at point 4, right part
    '''

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        self.block7 = BasicBlock(256, 512, 2,
                                 nn.Sequential(
                                     conv1x1(256, 512, 2),
                                     nn.BatchNorm2d(512)),
                                 self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.block8 = BasicBlock(512, 512, 1, None, self.groups, self.base_width,
                                 self.dilation, self._norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def forward(self, x):
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


####################
# Model Pair
####################
NetExit1Part1 = [NetExit1Part1L, NetExit1Part1R]
NetExit1Part2 = [NetExit1Part2L, NetExit1Part2R]
NetExit2Part1 = [NetExit2Part1L, NetExit2Part1R]
NetExit2Part2 = [NetExit2Part2L, NetExit2Part2R]
NetExit3Part1 = [NetExit3Part1L, NetExit3Part1R]
NetExit3Part2 = [NetExit3Part2L, NetExit3Part2R]
NetExit3Part3 = [NetExit3Part3L, NetExit3Part3R]
NetExit4Part1 = [NetExit4Part1L, NetExit4Part1R]
NetExit4Part2 = [NetExit4Part2L, NetExit4Part2R]
NetExit4Part3 = [NetExit4Part3L, NetExit4Part3R]
NetExit4Part4 = [NetExit4Part4L, NetExit4Part4R]
