import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import random
import numpy as np

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
SEED = 42

class BaseDataset:
    DATASETS = ["cifar10", "cifar100", "mnist", "imagenette"]

    def __init__(self, dataset_name, batch_size, train_split, normalize, torch_generator, sample_percent=None):
        assert dataset_name in self.DATASETS, f"Dataset {dataset_name} not supported. Supported datasets: {self.DATASETS}"
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_split = train_split
        self.normalize = normalize
        self.torch_generator = torch_generator
        self.sample_percent = sample_percent
        self.__set_image_size()
        self.__set_num_channels()
        self.__set_num_classes()
        self.__download_dataset()
        self.__split_dataset()
        if self.sample_percent is not None:
            self.__sample_dataset()
        self.log_dataset_info()

    def __set_image_size(self):
        if self.dataset_name in ["cifar10", "cifar100"]:
            self.image_size = 32
        elif self.dataset_name == "mnist":
            self.image_size = 28
        elif self.dataset_name == "imagenette":
            self.image_size = 224
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: {self.DATASETS}")
        
    def __set_num_classes(self):
        if self.dataset_name in ["cifar10", "mnist", "imagenette"]:
            self.num_classes = 10
        elif self.dataset_name == "cifar100":
            self.num_classes = 100
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: {self.DATASETS}")
        
    def __set_num_channels(self):
        if self.dataset_name in ["cifar10", "cifar100", "imagenette"]:
            self.num_channels = 3
        elif self.dataset_name == "mnist":
            self.num_channels = 1
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: {self.DATASETS}")

    def get_normalization_params(self):
        if self.dataset_name == "cifar10":
            return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        elif self.dataset_name == "cifar100":
            return [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        elif self.dataset_name == "mnist":
            return [0.1307], [0.3081]
        elif self.dataset_name == "imagenette":
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: {self.DATASETS}")

    def __get_train_transforms(self):
        if self.dataset_name in ["cifar10", "cifar100"]:
            return transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()] +
                ([transforms.Normalize(*self.get_normalization_params())] if self.normalize else [])
            )
        elif self.dataset_name == "mnist":
            return transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.ToTensor()] +
                ([transforms.Normalize(*self.get_normalization_params())] if self.normalize else [])
            )
        elif self.dataset_name == "imagenette":
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()] +
                ([transforms.Normalize(*self.get_normalization_params())] if self.normalize else [])
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: {self.DATASETS}")
        
    def __get_test_transforms(self):
        if self.dataset_name in ["cifar10", "cifar100", "mnist"]:
            return transforms.Compose([
                transforms.ToTensor()] +
                ([transforms.Normalize(*self.get_normalization_params())] if self.normalize else [])
            )
        elif self.dataset_name == "imagenette":
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()] +
                ([transforms.Normalize(*self.get_normalization_params())] if self.normalize else [])
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: {self.DATASETS}")

    def __download_dataset(self):
        print("Getting/downloading dataset...")
        if self.dataset_name == "cifar10":
            self.train_dataset = datasets.CIFAR10('./data', download=True, train=True, transform=self.__get_train_transforms())
            self.validation_dataset = datasets.CIFAR10('./data', download=True, train=True, transform=self.__get_test_transforms())
            self.test_dataset = datasets.CIFAR10('./data', download=True, train=False, transform=self.__get_test_transforms())
        elif self.dataset_name == "cifar100":
            self.train_dataset = datasets.CIFAR100('./data', download=True, train=True, transform=self.__get_train_transforms())
            self.validation_dataset = datasets.CIFAR100('./data', download=True, train=True, transform=self.__get_test_transforms())
            self.test_dataset = datasets.CIFAR100('./data', download=True, train=False, transform=self.__get_test_transforms())
        elif self.dataset_name == "mnist":
            self.train_dataset = datasets.MNIST('./data', download=True, train=True, transform=self.__get_train_transforms())
            self.validation_dataset = datasets.MNIST('./data', download=True, train=True, transform=self.__get_test_transforms())
            self.test_dataset = datasets.MNIST('./data', download=True, train=False, transform=self.__get_test_transforms())
        elif self.dataset_name == "imagenette":
            self.train_dataset = datasets.Imagenette('./data', download=True, split='train', 
                                                     transform=self.__get_train_transforms(), size="320px")
            self.validation_dataset = datasets.Imagenette('./data', download=True, split='train',
                                                        transform=self.__get_test_transforms(), size="320px")
            self.test_dataset = datasets.Imagenette('./data', download=True, split='val',
                                                     transform=self.__get_test_transforms(), size="320px")
            
    def __sample_dataset(self):
        # samples a percentage of the dataset for each split (train, valid, test)
        num_train_samples = int(len(self.train_dataset) * self.sample_percent)
        num_val_samples = int(len(self.validation_dataset) * self.sample_percent)
        num_test_samples = int(len(self.test_dataset) * self.sample_percent)

        self.train_dataset = Subset(self.train_dataset, random.sample(range(len(self.train_dataset)), num_train_samples))
        self.validation_dataset = Subset(self.validation_dataset, random.sample(range(len(self.validation_dataset)), num_val_samples))
        self.test_dataset = Subset(self.test_dataset, random.sample(range(len(self.test_dataset)), num_test_samples))
            
    def __split_dataset(self):
        # Split the dataset into training and validation sets
        train_size = int(len(self.train_dataset) * self.train_split)
        val_size = len(self.train_dataset) - train_size
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(self.train_dataset)), [train_size, val_size], generator=self.torch_generator)
        self.train_dataset = Subset(self.train_dataset, train_indices)
        self.validation_dataset = Subset(self.validation_dataset, val_indices)

    def create_dataloaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=self.torch_generator)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=self.torch_generator)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=self.torch_generator)
    
    def get_train_dataloader(self):
        if self.train_loader is None:
            raise ValueError("Dataloaders not created. Call create_dataloaders() first.")
        return self.train_loader
    
    def get_validation_dataloader(self):
        if self.validation_loader is None:
            raise ValueError("Dataloaders not created. Call create_dataloaders() first.")
        return self.validation_loader
    
    def get_test_dataloader(self):
        if self.test_loader is None:
            raise ValueError("Dataloaders not created. Call create_dataloaders() first.")
        return self.test_loader

    def log_dataset_info(self):
        print(f"Dataset: {self.dataset_name}")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.validation_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Normalization: {self.normalize}")

class AttackDataset(Dataset):
    def __init__(self, adv_images, org_images, model_idxs):
        self.adv_images = adv_images
        self.org_images = org_images
        self.model_idxs = model_idxs

    def __len__(self):
        return len(self.adv_images)

    def __getitem__(self, idx):
        image = self.adv_images[idx]
        model_idx = self.model_idxs[idx]
        label = self.org_images[idx]

        return image, label, model_idx

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_pgd = transforms.Compose([
    transforms.ToTensor()
])

transform_base_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_base_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cifar10_loader_resnet(device, batch_size, transform, torch_generator, train=False):
    dataset = datasets.CIFAR10('../data', download=True, train=train,
                          transform=transform)
    # if we are using cpu, take only subset of dataset
    if device == torch.device('cpu'):
        num_samples = len(dataset) // 50
        random_indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, random_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, worker_init_fn=seed_worker, generator=torch_generator)
    return dataloader
