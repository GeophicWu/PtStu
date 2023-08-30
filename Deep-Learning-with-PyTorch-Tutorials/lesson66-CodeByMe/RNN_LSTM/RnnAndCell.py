import torch
from torch import nn, optim
from torch import functional as F
from torchvision import transforms

import numpy as np


def simple_data(num_time_steps):
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start+10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps-1,1)

    print(start)
    print(time_steps)
    print(data.shape)
    print(x.shape)
    listm = [1, 2,3]
    print(listm[:-1])
    x = torch.tensor([[2,3,4],[1,2,3]])
    e = x.data.numpy().ravel()[1]
    print(e)


def main():
    simple_data(50)


if __name__ == '__main__':
    main()
