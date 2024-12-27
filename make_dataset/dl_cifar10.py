import torch
import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# データセットのダウンロード
def download_cifar10():
    root = "./CIFAR10"
    if os.path.exists(root):
        print("Already exists.")
    else:
        torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=root, train=False, download=True)

def download_mnist():
    root = "./CIFAR10/MNIST"
    if os.path.exists(root):
        print(root)
        print("Already exists.")
    else:
        torchvision.datasets.MNIST(root=root, train=True, download=True)
        torchvision.datasets.MNIST(root=root, train=False, download=True)


def download_svhn():
    root = "./SVHN"
    if os.path.exists(root):
        print(root)
        print("Already exists.")
    else:
        torchvision.datasets.SVHN(root=root, split="train", download=True)
        torchvision.datasets.SVHN(root=root, split="test", download=True)

if __name__ == "__main__":
    download_svhn()