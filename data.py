import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=1000):
    CIFAR10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)) # Normalize with mean and std of the CIFAR-10 dataset
                    ])),
        batch_size=batch_size,
        shuffle=True)
    CIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)) # Normalize with mean and std of the CIFAR-10 dataset
                    ])),
        batch_size=batch_size,
        shuffle=True)
    return CIFAR10_train_loader, CIFAR10_test_loader