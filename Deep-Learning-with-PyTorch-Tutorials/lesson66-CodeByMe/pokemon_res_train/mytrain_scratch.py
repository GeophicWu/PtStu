import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from mypokemon import Pokemon
from mypokemon import root_dir
from resnet import ResNet18

"""
Pokemon 数据集 分类训练-自定义的resnet18
"""
batch_size = 32
lr = 1e-3
epochs = 10
poto_size = 224
MODE_TRAIN = 'train'
MODE_VAL = 'val'
MODE_TEST = 'test'
MODE_FILE = 'best_model.mdl'

device = torch.device('cuda:0')
torch.manual_seed(1234)

train_db = Pokemon(root_dir, poto_size, mode=MODE_TRAIN)
val_db = Pokemon(root_dir, poto_size, mode=MODE_VAL)
test_db = Pokemon(root_dir, poto_size, mode=MODE_TEST)

train_loader = DataLoader(train_db, shuffle=True, batch_size=batch_size, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batch_size, num_workers=2)

viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)

        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():
    model = ResNet18(5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            print('eval acc:', val_acc, ' epoch:', best_epoch)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), MODE_FILE)

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load(MODE_FILE))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()
