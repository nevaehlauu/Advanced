"""SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 模型的具体内容 ----------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class ReversePreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ReversePreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.ConvTranspose1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, input_size, num_classes):
        super(SENet, self).__init__()
        self.model_type = "general"  # general 或者 seq2seq
        self.in_planes = 64

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_no_down = self._make_layer(block, 512, num_blocks[3], stride=1)  # 加一个，256 --> 512 的，但是不减小尺寸，加深特征提取
        self.in_planes = 256
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.layer5 = self._make_layer(block, 1024, num_blocks[3], stride=2)  # 再来一个

        self.embedding = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),

            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     x0 = self.layer4(out)
    #     # out = F.avg_pool1d(out, 4)
    #     x1 = F.adaptive_avg_pool1d(x0, 1)  # 改成自适应
    #     x1 = x1.view(x1.size(0), -1)
    #     x1 = self.linear(x1)
    #     # return x0, x1
    #     return x1
    

    def forward(self, x):
        """
        作为原型网络的输出，这里不需要经过分类器
        """
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = self.embedding(x)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1) # batchsize * hidden_size，这里hidden_size是512，也就是输出维度
        # print("out.shape-----------", out.shape)
        return out


def SENet18(in_channels, classes):
    return SENet(PreActBlock, [2, 2, 2, 2], in_channels, classes)


def ReverserSENet18(input_size, output_size):
    return SENet(ReversePreActBlock, [2, 2, 2, 2], input_size, output_size)


def test():
    reverser = ReverserSENet18(1, 1)
