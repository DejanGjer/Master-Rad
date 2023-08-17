import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import random

class AttackDataset(Dataset):
    def __init__(self, adv_images, org_images):
        self.adv_images = adv_images
        self.org_images = org_images

    def __len__(self):
        return len(self.adv_images)

    def __getitem__(self, idx):
        image = self.adv_images[idx]
        label = self.org_images[idx]

        return image, label

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_pgd = transforms.Compose([
    transforms.ToTensor()
])

def cifar10_loader_resnet(device, batch_size, transform, train=False, split=None):
    dataset = datasets.CIFAR10('../data', download=True, train=train,
                          transform=transform)
    # if we are using cpu, take only subset of dataset
    if device == torch.device('cpu'):
        num_samples = len(dataset) // 1000
        random_indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, random_indices)

    # if split is not None:
    #     split_datasets = random_split(dataset, split)
    #     dataloaders = []
    #     for split_dataset in split_datasets:
    #         dataloaders.append(DataLoader(split_dataset, batch_size=batch_size, shuffle=train))
    #     return dataloaders
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader
