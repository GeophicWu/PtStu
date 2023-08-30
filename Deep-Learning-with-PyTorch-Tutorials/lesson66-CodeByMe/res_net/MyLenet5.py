import torch
import torch.nn as nn
import torch.functional as F


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        # flatten
        tem = torch.randn(5, 3, 160, 160)
        out = self.conv_unit(tem)
        print("Test x shape after convolution {}".format(out.shape))
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)

        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_unit(x)
        # flatten
        x = x.view(batch_size, -1)
        logits = self.fc_unit(x)
        return logits


def main():
    net = Lenet5()

    tem = torch.randn(5, 3, 320, 320)
    out = net(tem)
    print("Test x shape after convolution {}".format(out.shape))


if __name__ == '__main__':
    main()
