import torch
from torch import nn

print("Hello GeophicWu")
print(torch.cuda.is_available())
print(torch.cuda)
print(torch.__version__)


def envs_torchtext():
    # from torchtext.legacy.data import Field
    import torchtext
    print(torchtext.__version__)
    from torchtext.legacy import data, datasets

    print('GPU:', torch.cuda.is_available())

    torch.manual_seed(123)

    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    print('len of train data:', len(train_data))
    print('len of test data:', len(test_data))

    print(train_data.examples[15].text)
    print(train_data.examples[15].label)
    #


def view_shape():
    data = torch.randn((3, 2, 2))
    print(data.shape)
    d = data.view(6, -1)
    print(d.shape)


def relu_true():
    x = torch.tensor([-1, 0, 1], dtype=torch.float32)
    relu = nn.ReLU()

    print("Original tensor:", x)

    # Using inplace=False (default)
    result_default = relu(x)
    print("ReLU inplace=False (default) result:", result_default)
    print("X value after inplace=False, {}".format(x))

    # Using inplace=True
    relu_inplace = nn.ReLU(True)
    result_inplace = relu_inplace(x)

    print("ReLU inplace=True result:", result_inplace)
    print("X value after inplace=True, {}".format(x))


def main():
    # envs_torchtext()
    # view_shape()
    relu_true()
    pass


if __name__ == '__main__':
    main()
