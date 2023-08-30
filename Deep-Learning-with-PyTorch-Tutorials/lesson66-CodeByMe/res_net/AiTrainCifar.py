import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from visdom import Visdom
from GlobalSet import CIFAR_10_DATA
from AiResNet import ResNet18

"""
Ai修改过后的残差训练。
"""


def main():
    batch_size = 128
    learning_rate = 0.001
    epochs = 5
    global_step = 0
    drop_prob = 0.2

    # milestones = [20, 40, 60, 80]  # Epochs at which to decrease learning rate
    milestones = list(range(30, 200, 30))  # Epochs at which to decrease learning rate

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cifar_train = datasets.CIFAR10(CIFAR_10_DATA, True, transform=transform, download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10(CIFAR_10_DATA, False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda:0')
    net = ResNet18(drop_prob=drop_prob).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criteon = nn.CrossEntropyLoss().to(device)

    # viz = Visdom()
    # viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc', legend=['loss', 'acc']))

    for i in range(epochs):
        net.train()
        loss_list = []

        for index_id, (datas, labels) in enumerate(cifar_train):
            datas, labels = datas.to(device), labels.to(device)
            logits = net(datas)
            loss_train = criteon(logits, labels)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            global_step += 1
            loss_list.append(loss_train.item())

            if global_step % 100 == 0:
                print("Epochs: {}\tStep: {}\tLoss: {:.6f}".format(i, global_step, np.mean(np.array(loss_list))))

        scheduler.step()

        net.eval()
        with torch.no_grad():
            total_num = 0
            total_acc = 0
            for data_t, label_t in cifar_test:
                data_t, label_t = data_t.to(device), label_t.to(device)
                logits = net(data_t)
                pred = logits.argmax(dim=1)
                total_acc += torch.eq(pred, label_t).float().sum()
                total_num += data_t.size(0)

            print("Epoch: {}\tAccuracy: {:.3f}".format(i, total_acc / total_num))

        # viz.line([[np.mean(np.array(loss_list)), total_acc.detach().cpu() / total_num]],
        #          [global_step], win='test', update='append')


if __name__ == '__main__':
    main()
