import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import functional as F


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def plt_show():
    x = torch.arange(-6, 6, .1)
    y = torch.arange(-6, 6, .1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])
    print("\n先熟悉下numpy的矩阵乘法和加法")
    print("乘法和点乘还是有很大的去别的啊")
    x1 = np.array([[1, 2, 3], [0, 0, 0]])
    x2 = np.array([[0, 0, 0], [1, 2, 3]])
    y1 = x1 * x2
    print(np.array(y1).shape)
    fig = plt.figure("himellblau")
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


# plt_show()


def run_gradient_descent():
    x = torch.tensor([4., 0.], requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.001)
    for step in range(20000):
        pred = himmelblau(x)
        optimizer.zero_grad()
        pred.backward()
        optimizer.step()
        if step % 2000 == 0:
            print("step{}:x={},f(x)={}".format(step, x.tolist(), pred.item()))


# run_gradient_descent()


def run_cross_entropy():
    x = torch.randn(1, 784)
    w = torch.randn(10, 784)
    logits = x @ w.t()
    pred = F.softmax(logits, dim=1)
    pred_log = torch.log(pred)
    res = F.nll_loss(pred_log, torch.tensor([3]))
    print("\n多分类预测交叉熵：{}".format(res.item()))
    res_ce = F.cross_entropy(logits, torch.tensor([3]))
    print("多分类预测交叉熵：{}".format(res_ce.item()))


# run_cross_entropy()



