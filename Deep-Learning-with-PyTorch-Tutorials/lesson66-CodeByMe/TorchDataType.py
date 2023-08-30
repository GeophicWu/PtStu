import torch
from torch.nn import functional as F

data = torch.randn(6, 2)
print("data type: {}".format(data.type()))
print("check data type---cuda data :{}".format(isinstance(data, torch.cuda.FloatTensor)))
# data = data.cuda()
print("check data type---cuda data :{}".format(isinstance(data, torch.cuda.FloatTensor)))
print("dim : {}".format(data.dim()))

import numpy as np

rand_int = torch.randint(1, 10, [3, 2])
print(rand_int)
lin_space = torch.linspace(0, 1, 10)
print(lin_space)
print(data)
data_index = data.index_select(0, torch.tensor([0, 1, 2]))
print(data_index)
print(torch.arange(2, 8))

data_randn = torch.randn(7, 3)
d_mask = data_randn.ge(0.5)
data_h = torch.masked_select(data_randn, d_mask)
print(data_h)
print(data_h.shape)


def run_torch_expand():
    data_expand = torch.randn(4, 42, 1, 1)
    print("\nshape before expand: {}".format(data_expand.shape))
    data_expand = data_expand.expand(-1, -1, -1, 4)
    print("shape after expand: {}".format(data_expand.shape))


run_torch_expand()


def run_torch_transpose():
    data = torch.randn(4, 3, 32, 32)
    data1 = data.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)
    data2 = data.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)
    # print(torch.eq(data1,data2).shape) # [4,3,32,32]
    print("data1-data：equal???:{}".format(torch.all(torch.eq(data1, data))))
    print("data2-data：equal???:{}\n".format(torch.all(torch.eq(data2, data))))


run_torch_transpose()


def run_torch_clamp():
    print("Test torch clamp func")
    grad = torch.rand(3, 4) * 30 - torch.tensor(15)
    # grad = torch.rand(3, 4) * 15
    print(grad)
    print(grad.median())
    print(grad.max())
    g1 = grad.clamp(10)
    print(g1)
    g2 = grad.clamp(0, 10)
    print(g2)
    # grad.kthvalue()

    # torch.clamp()


run_torch_clamp()


def run_torch_activate():
    z = torch.linspace(-100, 100, 10)
    sig_val1 = F.sigmoid(z)
    sig_val2 = torch.sigmoid(z)
    print("\n激活函数：")
    print(sig_val1)
    print(torch.all(torch.eq(sig_val2, sig_val1)))


run_torch_activate()


def run_mse_auto_grad():
    x = torch.ones(1)
    w = torch.full([1], 2.)
    w.requires_grad_()
    # w = torch.tensor([2.0], requires_grad=True)

    mse = F.mse_loss(w * x, torch.ones(1))
    # g = torch.autograd.grad(mse, [w])
    # print("\n自动对loss和参数w求导：{}".format(g))
    mse.backward()
    print("\n自动对loss和参数w求导：{}".format(w.grad))
    print("w-grad-norm：{}".format(w.grad.norm()))


run_mse_auto_grad()


def run_soft_max():
    y = torch.tensor([2., 1., 0.1], requires_grad=True)

    s_y = F.softmax(y, dim=0)
    print("\nSoft version of max{}".format(s_y))
    print(s_y.shape)
    s_y.backward(torch.tensor([1, 0, 0]), retain_graph=True)
    # s_y.backward(torch.tensor([0, 1, 0]), retain_graph=True)
    # s_y.backward(torch.tensor([0, 0, 1]), )
    print("Print grad of p1:{}".format(y.grad))
    y.grad.zero_()
    s_y.backward(torch.tensor([0.0, 1.0, 0.]), retain_graph=True)
    print("Print grad of p1:{}".format(y.grad))
    y.grad.zero_()
    s_y.backward(torch.tensor([0.0, 0.0, 1.]))
    print("Print grad of p1:{}".format(y.grad))

    # tg1 = torch.autograd.grad(s_y[0], [y], retain_graph=True)
    # tg2 = torch.autograd.grad(s_y[2], [y])
    # print("Print grad of p1:{}".format(tg1))
    # print("Print grad of p2:{}".format(tg2))


run_soft_max()


def run_signal_perceptron():
    print("\n Run simple perceptron:")
    x = torch.randn(1, 10)
    w = torch.randn(1, 10, requires_grad=True)
    o = F.sigmoid(x @ w.t())
    loss = F.mse_loss(o, torch.ones(1, 1))
    loss.backward()
    print("First level of the hidden w grad:{}".format(list(w.grad)))
    print("First level of the hidden w grad:{}".format(w.grad.norm()))
    print("\n Run multi perceptron:")
    x = torch.randn(1, 10)
    w = torch.randn(2, 10, requires_grad=True)
    o = F.sigmoid(x @ w.t())
    loss = F.mse_loss(o, torch.ones(1, 2))
    loss.backward()
    print("First level of the hidden w grad:{}".format(list(w.grad.shape)))
    print("First level of the hidden w grad:{}".format(w.grad.norm()))


run_signal_perceptron()

