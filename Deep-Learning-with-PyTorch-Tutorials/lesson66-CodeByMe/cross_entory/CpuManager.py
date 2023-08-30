import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

"""
    演示在CPU上训练mnist数据集
"""


def cuda_count():
    # device = torch.device('cuda')
    num = torch.cuda.device_count()
    print("Person Computer has {} chunk Gpu ".format(num))


cuda_count()

# train on CPU
batch_size = 200
lr = 0.01
epochs = 20

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('Class9/mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('Class9/mnist_data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


device = torch.device('cpu')  # GPU 设备nvidia-smi
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)
criteon = nn.CrossEntropyLoss().to(device)

import time

time_start = time.time()
for epoch in range(epochs):
    time_epoch = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  len_dataset:{},len_tr_loader:{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), len(train_loader.dataset), len(train_loader)
            ))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        test_loss += criteon(logits, target)
        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set:Average loss: {:.4f}, Accuracy :{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    print("Epoch: {}---cost time:{} sec\n".format(epoch, (time.time() - time_epoch) / 1.))

time_end = time.time()

print("Total train test cost time:{} sec".format((time_end - time_start) / 1.0))
