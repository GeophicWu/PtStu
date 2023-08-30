import numpy as np
import torch
from visdom import Visdom
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from GlobalSet import CIFAR_10_DATA

# from mylenet5 import Lenet5
from myresnet import ResNet18


def main():
    batch_size = 128

    cifar_train = datasets.CIFAR10(CIFAR_10_DATA, True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10(CIFAR_10_DATA, False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = next(iter(cifar_train))
    print('x:', x.shape, 'label:', label.shape)
    # viz = Visdom()
    # viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc', legend=['loss', 'acc']))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    # net = Lenet5().to(device)
    net = ResNet18().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.01)
    criteon = nn.CrossEntropyLoss().to(device)
    print(net)
    epochs = 5
    global_step = 0
    import time
    for i in range(epochs):
        # time_start = time.time()
        net.train()
        loss_list = []
        for index_id, (datas, labels) in enumerate(cifar_train):
            time_st = time.time()
            datas, labels = datas.to(device), labels.to(device)
            logits = net(datas)
            loss_train = criteon(logits, labels)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            global_step += 1
            loss_list.append(loss_train.item())
            if global_step % 100 == 0:
                print("Epochs:\t{}\tstep:{:0>5}\ttime:{:0.6f}\tloss:{:0.6f}".format(i, global_step,
                                                                                    (time.time() - time_st)*100.,
                                                                                    np.mean(np.array(loss_list))))

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
            print("Epoch:{}\tAcc:{:0.3f}".format(i, total_acc / total_num))


        # print(type(total_acc))
        # print(total_acc.device)
        # print(type(loss_list))
        # print(loss_list.device)

        # viz.line([[np.mean(np.array(loss_list)), total_acc.detach().cpu() / total_num]],
        #          [global_step], win='test', update='append')


if __name__ == '__main__':
    main()
