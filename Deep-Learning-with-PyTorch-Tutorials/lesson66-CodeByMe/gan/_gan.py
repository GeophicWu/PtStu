import torch
import numpy as np
from torch import nn, optim, autograd
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random
import visdom

h_dim = 400
bs = 512
viz = visdom.Visdom()
device = torch.device("cuda")

"""
Gan W_gan 网络 训练 生成高斯分布。
"""


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, x):
        output = self.net(x)
        return output


def data_generator():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = [(x * scale, y * scale) for x, y in centers]
    while True:
        dataset = []
        for i in range(bs):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414
        yield dataset


def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).to(device)  # [16384, 2]
        disc_map = D(points).cpu().numpy()  # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()

    # draw samples
    with torch.no_grad():
        z = torch.randn(bs, 2).to(device)  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]
    plt.scatter(xr.cpu()[:, 0], xr.cpu()[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


def train_gan():
    # global loss_D
    torch.manual_seed(23)
    np.random.seed(23)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CPU if CUDA is not available
    print("GPU is available:{}".format(torch.cuda.is_available()))
    G = Generator().to(device)
    D = Discriminator().to(device)
    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))
    data_iter = data_generator()
    print("Batch data shape:{}".format(next(data_iter).shape))
    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))
    # 1.train Discriminator for k steps,first
    for epoch in range(5000):
        for _ in range(5):
            x = next(data_iter)
            xr = torch.from_numpy(x).to(device)
            # [b]
            pred = D(xr)
            # J(theta)=-Wassertein distance = -(f_D - f_D(gx))
            lossr = -pred.mean()
            # random virtual data randn,shape[b, 2]
            z = torch.randn(bs, 2).to(device)
            # such as tf.stop_gredient(), train D net there is no need to calculate gradient on G
            xf = G(z).detach()
            predf = D(xf)
            lossf = predf.mean()
            loss_D = lossr + lossf
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # train Generator
        # For G net purpose, it finally wanna to create a similar x on random,blank,or other virtual data
        # So train g on
        z = torch.randn(bs, 2).to(device)
        xf = G(z)
        predf = D(xf)
        loss_G = -predf.mean()
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            # viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print("Epoch:{} --- D loss:{} --- G loss {}".format(epoch, loss_D.item(), loss_G.item()))
            generate_image(D, G, xr, epoch)


def main():
    train_gan()


if __name__ == '__main__':
    main()
