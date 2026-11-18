"""Original NTMP MLP and option-A ResNet-20, reduced to supported datasets."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(28 * 28, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer2 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer3 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer4 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer5 = nn.Linear(300, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(module.weight)


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, 1)
        self.layer2 = self._make_layer(32, 3, 2)
        self.layer3 = self._make_layer(64, 3, 2)
        self.linear = nn.Linear(64, 1)
        self.apply(_weights_init)

    def _make_layer(self, planes: int, blocks: int, first_stride: int) -> nn.Sequential:
        layers = []
        for stride in [first_stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        return self.linear(out.view(out.size(0), -1))


def build_model(dataset: str) -> nn.Module:
    key = dataset.lower().replace("-", "")
    if key in {"mnist", "fashionmnist"}:
        return MLP()
    if key in {"cifar10", "svhn"}:
        return ResNet20()
    raise ValueError(f"unsupported dataset: {dataset}")
