import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import time
from datetime import datetime as dt
from torchvision import datasets, transforms
import copy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import config
from fgsm_attack import FGSMDataset, run_fgsm, test_fgsm
from utils import load_model, save_model, create_save_directories, plot_loss_history, set_compute_device
from dataset import cifar10_loader_resnet, transform_train, transform_test, transform_pgd
from denoiser import train_denoiser, test_denoiser
from unet import UNet
from resnet18 import ResNet, BasicBlock

torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

def create_fgsm_dataset(fgsm_loader, model, device):
    start_time = time.time()
    # Create detaset from fgsm attacked images
    print("Creating FGSM dataset...")
    adv_images, org_images = run_fgsm(model, device, fgsm_loader, config.epsilons)
    fgsm_train_dataset = FGSMDataset(adv_images, org_images)
    # split training dataset into training and validation
    fgsm_train_dataset, fgsm_validation_dataset = torch.utils.data.random_split(fgsm_train_dataset, 
                                                    [config.train_split, config.validation_split])
    fgsm_train_loader = DataLoader(fgsm_train_dataset, batch_size=config.batch_size, shuffle=True)
    fgsm_validation_loader = DataLoader(fgsm_validation_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Training fgsm set size: {len(fgsm_train_loader.dataset)}")
    print(f"Validation fgsm set size: {len(fgsm_validation_loader.dataset)}")
    print('--- Total time: %s seconds ---' % (time.time() - start_time))

    return fgsm_train_loader, fgsm_validation_loader


if __name__ == "__main__":
    # Set device
    device = set_compute_device()

    print("Configuration details:")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Bilinear: {config.bilinear}")
    print(f"Learn noise: {config.learn_noise}")
    print(f"Loss type: {config.loss}")
    print("---------------------------------")
    model_normal = load_model(config.model_path, device)
    print(f"Loaded model {model_normal.net_type}")

    loader = cifar10_loader_resnet
    train_loader = loader(device, config.batch_size, transform_train, train=True)
    test_loader = loader(device, config.batch_size, transform_test)
    fgsm_loader = loader(device, 1, transform_train, train=True)
    pgd_loader = loader(device, config.batch_size, transform_pgd, train=True)

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"FGSM set size: {len(fgsm_loader.dataset)}")

    fgsm_train_loader, fgsm_validation_loader = create_fgsm_dataset(fgsm_loader, model_normal, device)

    # Train denoising model

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=3, bilinear=config.bilinear, learn_noise=config.learn_noise)
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    start_time = time.time()

    metrics = {
        'train_loss': [],
        'validation_loss': []
    }
    # training loop
    for epoch in range(1, config.epochs + 1):
        train_losses = train_denoiser(model, fgsm_train_loader, optimizer, config.loss, device, 
                                      defended_model=None if config.loss == "pgd" else model_normal)
        print(f"Average training loss (epoch {epoch}): {np.mean(train_losses)}")
        validation_losses = test_denoiser(model, fgsm_validation_loader, config.loss, device,
                                          defended_model=None if config.loss == "pgd" else model_normal)
        print(f"Average validation loss (epoch {epoch}): {np.mean(validation_losses)}")
        metrics["train_loss"].append(np.mean(train_losses))
        metrics["validation_loss"].append(np.mean(validation_losses))

    save_dir = create_save_directories(config.save_root_path)
    save_model(model, os.path.join(save_dir, 'unet_denoiser.pt'))
    plot_loss_history(metrics, save_dir)
    # save configuration file
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        f.write(str(config.__dict__))
    # load unet model

    model.eval()
    # Run test for each epsilon value
    average_improvement = 0
    for epsilon in config.epsilons:
        test_fgsm(model_normal, device, train_loader, epsilon, dataset_type="train")
        test_fgsm(model_normal, device, train_loader, epsilon, denoiser=model, dataset_type="train")
        _, _, acc1 = test_fgsm(model_normal, device, test_loader, epsilon, dataset_type="test")
        _, _, acc2 = test_fgsm(model_normal, device, test_loader, epsilon, denoiser=model, dataset_type="test")
        average_improvement += acc2 - acc1
    print(f"Average imprevement: {100 * average_improvement / len(config.epsilons)}%")