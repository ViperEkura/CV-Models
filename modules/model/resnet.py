import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
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
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1
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
    
    configs: Dict[str, Tuple[nn.Module, List[int], List[int]]] = {
        'resnet18': (BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], 1),
        'resnet34': (BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], 1),
        'resnet50': (Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512], 4),
        'resnet101': (Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512], 4),
        'resnet152': (Bottleneck, [3, 8, 36, 3], [64, 128, 256, 512], 4),
    }

    def __init__(self, version, in_channel, out_dim=None):
        super(ResNet, self).__init__()
        assert version in self.configs
        block_type, block_list, channel_list, expansion = self.configs[version]
        self.expansion = expansion
        self.out_dim = out_dim
        self.in_planes = 64

        self.conv = nn.Conv2d(in_channel, self.in_planes, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
    
        layer_list = []
        for i in range(len(channel_list)):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(block_type, channel_list[i], block_list[i], stride)
            layer_list.append(layer)
        
        self.layers = nn.Sequential(*layer_list)
        if self.out_dim is not None:
            self.linear = nn.Linear(self.in_planes, out_dim)

    def _make_layer(self, block_type, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = block_type(self.in_planes, planes, stride)
            layers.append(block)
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layers(out)
        
        if self.out_dim is not None:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
        return out