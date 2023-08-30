import torch
from torch import nn
from torch.nn import functional as F

"""
Ai 修改过后的残差网络。
"""


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, drop_prob=0.0):  # Added drop_prob parameter
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # Added dropout layer
        self.dropout = nn.Dropout(drop_prob)

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.extra(x)
        out = self.dropout(out)  # Apply dropout

        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, drop_prob=0.0):  # Added drop_prob parameter
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.blk1 = ResBlk(64, 128, stride=2, drop_prob=drop_prob)  # Added drop_prob parameter
        self.blk2 = ResBlk(128, 256, stride=2, drop_prob=drop_prob)  # Added drop_prob parameter
        self.blk3 = ResBlk(256, 512, stride=2, drop_prob=drop_prob)  # Added drop_prob parameter
        self.blk4 = ResBlk(512, 512, stride=2, drop_prob=drop_prob)  # Added drop_prob parameter

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.maxpool(x)     # Apply max pooling
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x
