import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

batch_size = 512
# step1 load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False
)

x, y = next(iter(train_loader))
print(x.shape, y.shape)


# plot_image(x, y, "image figure")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


train_losses = []
net = Net()
# optimizer: function for accurate gradient with params [w,b]
# [w1, b1, w2, b2, w3, b3] in net model
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# train step
def run_train():
    for epoch in range(3):
        print("Train step: {}".format(epoch))
        for batch_idx, (x, y) in enumerate(train_loader):
            # print(x.shape, y.shape)
            # data shape
            # x.shape:torch.Size([512, 1, 28, 28]) y.shape:torch.Size([512])
            # transform data shape to fit net requirement
            # [b, 1, 28, 28]=>[b, feature]
            x = x.view(x.size(0), 28 * 28)
            out = net(x)
            y_one_hot = one_hot(y)
            loss = F.mse_loss(out, y_one_hot)
            optimizer.zero_grad()
            loss.backward()
            # w' = w - lr*grad
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 10 == 0:
                print("Batch_index: {}--{}--{}".format(epoch, batch_idx, loss.item()))


run_train()


def run_test():
    total_correct = 0
    for x, y in test_loader:
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        # out.shape:[b,10]=>pred:[b,1]
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct
    total_num = len(test_loader.dataset)
    acc = total_correct / total_num
    print("Test acc :{}".format(acc))


run_test()
# plot_curve(train_losses)


# recognize figure number by trained network
def run_predict():
    print("kaishi predict")
    x, y = next(iter(test_loader))
    out = net(x.view(x.size(0), 28 * 28))
    pred = out.argmax(dim=1)
    # show picture of x, and title it with pred value,
    # so you can see whether it is right to recognize handwrite number
    plot_image(x, pred, "test")

run_predict()
print("hand print figure recognize")
