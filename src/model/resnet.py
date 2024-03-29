"""
Modified from pytorch's resnet implementation for spectrogram input.
"""
from typing import Type, Any, Callable, Union, List, Optional
from PIL.Image import init

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from sklearn.metrics import f1_score

import pytorch_lightning as pl


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        padding: int = 1,
        kernel_size: Union[int, tuple] = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet18(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.layer_1 = self._make_layer(BasicBlock, inplanes=16, planes=32, stride=1)
        self.layer_2 = self._make_layer(
            BasicBlock,
            kernel_size=(5, 1),
            inplanes=32,
            planes=32,
            stride=2,
            padding=(2, 0),
        )
        self.layer_3 = self._make_layer(BasicBlock, inplanes=32, planes=64, stride=2)
        self.layer_4 = self._make_layer(BasicBlock, inplanes=64, planes=128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        padding: int = 1,
        kernel_size: Union[int, tuple] = 3,
        blocks: int = 2,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, padding, kernel_size, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, padding, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # conv2d 5x5, 16, stride 2
        # batch norm
        # relu
        # shape: (..., 16, 64, 108)
        # max pool 3x3, stride [1, 2]
        # shape: (..., 16, 64, 54)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer 1
        # block 2 layers, 3x3, 32
        # block 2 layers, 3x3, 32
        # shape: (..., 32, 64, 54)
        x = self.layer_1(x)

        # layer 2
        # downsample
        # shape: (..., 32, 32, 27)
        # block 2 layers, 5x1, 32
        # block 2 layers, 5x1, 32
        x = self.layer_2(x)

        # layer 3
        # downsample
        # shape: (..., 32, 16, 14)
        # block 2 layers, 3x3, 64
        # block 2 layers, 3x3, 64
        # shape: (..., 64, 16, 14)
        x = self.layer_3(x)

        # layer 4
        # down sample
        # shape: (..., 64, 8, 7)
        # block 2 layers, 3x3, 128
        # block 2 layers, 3x3, 128
        x = self.layer_4(x)

        # avg pool
        # shape: (..., 128, 1, 1)
        # n_classes fc
        # shape: (..., n_classes)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # out shape: (..., n_classes)
        return x


class resnet14(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.layer_1 = self._make_layer(BasicBlock, inplanes=16, planes=32, stride=1)
        self.layer_2 = self._make_layer(
            BasicBlock,
            kernel_size=(5, 1),
            inplanes=32,
            planes=32,
            stride=2,
            padding=(2, 0),
        )
        self.layer_3 = self._make_layer(BasicBlock, inplanes=32, planes=64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        padding: int = 1,
        kernel_size: Union[int, tuple] = 3,
        blocks: int = 2,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, padding, kernel_size, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, padding, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class resnet10(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.layer_1 = self._make_layer(BasicBlock, inplanes=16, planes=32, stride=1)
        self.layer_2 = self._make_layer(
            BasicBlock,
            kernel_size=(5, 1),
            inplanes=32,
            planes=32,
            stride=2,
            padding=(2, 0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        padding: int = 1,
        kernel_size: Union[int, tuple] = 3,
        blocks: int = 2,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, padding, kernel_size, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, padding, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class resnet6(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.layer_1 = self._make_layer(BasicBlock, inplanes=16, planes=32, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        padding: int = 1,
        kernel_size: Union[int, tuple] = 3,
        blocks: int = 2,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, padding, kernel_size, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, padding, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
