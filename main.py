import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import os
import shutil

import pandas as pd

import config
from resnet18 import ResNet, BasicBlock
from utils import load_model, save_model, create_save_directories, plot_loss_history, set_compute_device
from dataset import cifar10_loader_resnet, transform_train, transform_test, AttackDataset
from denoiser import train_denoiser, test_denoiser
from unet import UNet

from attacks import Attack, FGSMAttack, PGDAttack, OnePixelAttack

torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

def create_attack(attack_type, attack_params) -> Attack:
    if attack_type == "fgsm":
        return FGSMAttack("FGSM", **attack_params["fgsm"])
    elif attack_type == "pgd":
        return PGDAttack("PGD", **attack_params["pgd"])
    elif attack_type == "one_pixel":
        return OnePixelAttack("OnePixel", **attack_params["one_pixel"])
    else:
        raise ValueError("Invalid attack type")

def create_attack_dataset(attack, attack_loader, models):
    start_time = time.time()
    # Create detaset from fgsm attacked images
    print(f"Creating {attack} attacked dataset...")
    adv_images, org_images, model_idxs = [], [], []
    for i, model in enumerate(models):
        print(f"Running {attack} attack on {model.net_type}...")
        temp_adv_images, temp_org_images = attack.run_attack(model, attack_loader)
        adv_images.extend(temp_adv_images)
        org_images.extend(temp_org_images)
        model_idxs.extend([i for _ in range(len(temp_adv_images))])

    attack_train_dataset = AttackDataset(adv_images, org_images, model_idxs)
    # split training dataset into training and validation
    attack_train_dataset, attack_validation_dataset = random_split(attack_train_dataset, 
                                                    [config.train_split, config.validation_split])
    attack_train_loader = DataLoader(attack_train_dataset, batch_size=config.batch_size, shuffle=True)
    attack_validation_loader = DataLoader(attack_validation_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Training {attack} attacked dataset size: {len(attack_train_loader.dataset)}")
    print(f"Validation {attack} attacked dataset size: {len(attack_validation_loader.dataset)}")
    print('--- Total time: %s seconds ---' % (time.time() - start_time))

    return attack_train_loader, attack_validation_loader


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

    # Loading models needed for training
    attacked_models = []
    for model_path in config.train_model_paths:
        attacked_models.append(load_model(model_path, device))
        print(f"Loaded model {attacked_models[-1].net_type}")

    loader = cifar10_loader_resnet
    attack_loader = loader(device, config.batch_size, transform_train, train=True)
    test_loader = loader(device, config.batch_size, transform_test)

    print(f"Attack type: {config.attack_type}")
    attack = create_attack(config.attack_type, config.attack_params)
    attack_train_loader, attack_validation_loader = create_attack_dataset(
        attack, attack_loader, attacked_models
)
    print(f"Training set size: {len(attack_train_loader.dataset)}")
    print(f"Validation set size: {len(attack_validation_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

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
        train_losses = train_denoiser(model, attack_train_loader, optimizer, config.loss, device, 
                                      defended_models=attacked_models)
        print(f"Average training loss (epoch {epoch}): {np.mean(train_losses)}")
        validation_losses = test_denoiser(model, attack_validation_loader, config.loss, device,
                                          defended_models=attacked_models)
        print(f"Average validation loss (epoch {epoch}): {np.mean(validation_losses)}")
        metrics["train_loss"].append(np.mean(train_losses))
        metrics["validation_loss"].append(np.mean(validation_losses))

    save_dir, pgd_save_dir = create_save_directories(config.save_root_path, config.pgd_save_path if config.attack_type == "pgd" else None)
    save_model(model, os.path.join(save_dir, 'unet_denoiser.pt'))
    plot_loss_history(metrics, save_dir)
    # convert metrics to pandas dataframe and save it
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(save_dir, 'losses.csv'), index=False)
    # save configuration file
    config_save_path = os.path.join(save_dir, 'config.py')
    shutil.copyfile('config.py', config_save_path)

    # Loading models needed for testing
    test_models = []
    for model_path in config.test_model_paths:
        test_models.append(load_model(model_path, device))
        print(f"Loaded model {test_models[-1].net_type}") 
        test_models[-1].eval()

    model.eval()
    # Run test for each epsilon value
    total_average_improvement = 0
    # pandas dataframe to save all detailed results
    results = None
    averaged_results = pd.DataFrame(columns=['Model', 'Average Improvement'])

    for attacked_model in test_models:
        total_model_improvement = 0
        for attack_params in attack:
            result = None
            result1 = attack.test_attack(attacked_model, test_loader, **attack_params)
            results = pd.concat([results, result1], ignore_index=True) if results is not None else result1
            result2 = attack.test_attack(attacked_model, test_loader, denoiser_model=model, **attack_params)
            results = pd.concat([results, result2], ignore_index=True)
            total_model_improvement += (result2["Accuracy"] - result1["Accuracy"]).iloc[0]

        total_average_improvement += total_model_improvement / len(attack)
        averaged_results = pd.concat([averaged_results,
                                      pd.DataFrame({'Model': [attacked_model.net_type], 
                                      'Average Improvement': [total_model_improvement / len(attack)]})], 
                                      ignore_index=True)
        print(f"Average improvement for {attacked_model.net_type}: {100 * total_model_improvement / len(attack)}%")
    print(f"Average improvement for all models: {100 * total_average_improvement / (len(test_models))}%")
    # save results dataframes
    results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    averaged_results.to_csv(os.path.join(save_dir, 'averaged_results.csv'), index=False)