import csv
import glob
import os
import random
import time

import torch
import torchvision.datasets
import visdom
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            # for name in os.listdir(os.path.join(root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        self.images, self.labels = self.load_csv(r'pkmon.csv')
        if mode == "train":
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == "val":
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print("总图片数量：{},\n图片详情:\n{}".format(len(images), images))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for image in images:
                    name = image.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([image, label])
                print("Write into :{}".format(os.path.join(self.root, filename)))

        images_t, labels_t = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images_t.append(img)
                labels_t.append(label)
        assert len(images_t) == len(labels_t)
        # print('Load pokemon total data over,images number:{},labels number:{}'.format(len(images_t), len(labels_t)))
        return images_t, labels_t

    def __len__(self):
        return len(self.images)

    def denormalize(self, x):
        # x = (x^-mean)/std
        # x^ = x*std+mean
        # x shape[c,h,w], mean&std shape[c], so need to expand mean dim
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
        # print("Mean unsqueeze shape:{}".format(mean_tensor.shape))
        std_tensor = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
        return x * std_tensor + mean_tensor

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((int(1.25 * self.resize), int(1.25 * self.resize))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label


root_dir = r'D:\WorkSta\PtStu\data_source\pokemon\pokeman'


def poke_main():
    viz = visdom.Visdom()
    poke = Pokemon(root_dir, 64, 'val')
    x, y = next(iter(poke))
    viz.images(poke.denormalize(x), win="simple_x", opts=dict(title='sample_x'))
    print("Test single pokemon data shape x:{}-y:{}".format(x.shape, y.shape))
    print("Train data number:{}".format(len(poke.images)))
    loader = DataLoader(poke, batch_size=32, shuffle=True)
    for x, y in loader:
        print('Dataloader shape x:{}'.format(x.shape))
        print('Dataloader shape y:{}'.format(y.shape))
        viz.images(poke.denormalize(x), win='batch', nrow=8, opts=dict(title='batch_x'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
        time.sleep(10)


def folder_main():
    viz = visdom.Visdom()
    tf = transforms.Compose([
        # lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((int(1.25 * 64), int(1.25 * 64))),
        transforms.RandomRotation(15),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    poke = torchvision.datasets.ImageFolder(root_dir, transform=tf)
    loader = DataLoader(poke, batch_size=32, shuffle=True)
    print("Torchvision create dictionary itself:{}".format(poke.class_to_idx))
    for x, y in loader:
        print('Dataloader shape x:{}'.format(x.shape))
        print('Dataloader shape y:{}'.format(y.shape))
        viz.images(x, win='batch', nrow=8, opts=dict(title='batch_x'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
        time.sleep(10)


if __name__ == '__main__':
    # poke_main()
    folder_main()
