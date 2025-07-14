import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expandsion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expandsion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    configs = {
        'resnet18': (BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]),
        'resnet34': (BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512]),
        'resnet50': (Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512]),
        'resnet101': (Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512]),
        'resnet152': (Bottleneck, [3, 8, 36, 3], [64, 128, 256, 512]),
    }

    def __init__(self, version, in_channel, num_classes):
        super(ResNet, self).__init__()
        if version not in self.configs:
            raise ValueError(f"Unsupported ResNet version: {version}")
        block_type, num_blocks_list, num_channels_list = self.configs[version]
        self.in_planes = 64

        self.conv = nn.Conv2d(in_channel, num_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels_list[0])
        
        layer_list = [self._make_layer(block_type, num_channels_list[i], num_blocks_list[i], stride=1)
                  for i in range(len(num_channels_list))]
        self.layers = nn.Sequential(layer_list)
        
        self.linear = nn.Linear(num_channels_list[-1] * block_type.expansion, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block_type(self.in_planes, planes, stride))
            self.in_planes = planes * block_type.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out