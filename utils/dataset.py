import numpy as np
import torchvision
import torchvision.transforms as transforms

from utils.cutout import Cutout

class CIFAR10:
    def __init__(self, image_size=32):
        mean = np.array([0.49140082, 0.48215898, 0.44653094])
        std = np.array([0.24703225, 0.24348514, 0.26158785])

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(image_size, image_size // 8),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(image_size // 2)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)


class CIFAR100:
    def __init__(self, image_size=32):
        mean = np.array([0.50707578, 0.48655031, 0.44091914])
        std = np.array([0.26733428, 0.25643846, 0.27615047])

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(image_size, image_size // 8),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(image_size // 2)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.train_set = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)
        self.test_set = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

class TINYIMAGENET:
    def __init__(self, image_size=32):
        mean = np.array([0.48026012, 0.44807789, 0.39754902])
        std = np.array([0.25409287, 0.24558322, 0.26039191])

        train_transform = transforms.Compose([
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(int(image_size / 8 * 9)),  # simulating ResizedCrop of scale around 0.9, while also similar to imagenet's Resize(256) + CenterCrop 224
            torchvision.transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.train_set = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_transform)
        self.test_set = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)


