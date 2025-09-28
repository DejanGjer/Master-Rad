import os
import sys
from pathlib import Path

# Add the code directory to Python path
code_path = os.path.join(os.path.dirname(__file__), 'denoised_smoothing', 'code')
sys.path.append(code_path)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
import time
import os

import pandas as pd

import config_test as config
from resnet18 import ResNet, BasicBlock
from utils import load_model, save_model, create_save_directories, plot_loss_history, set_compute_device, save_config_file
from dataset import cifar10_loader_resnet, transform_train, transform_test, AttackDataset, BaseDataset, seed_worker
from denoiser import train_denoiser, test_denoiser
from unet import UNet
from attacks import Attack, FGSMAttack, RFGSMAttack, PGDAttack, OnePixelAttack, PixleAttack, SquareAttack

from denoised_smoothing.code.architectures import get_architecture

# import sys
# from pathlib import Path
# import importlib.util

# # Path to architectures.py
# module_path = Path("denoised-smoothing/code/architectures.py").resolve()
# module_name = "denoised_smoothing_architectures"

# # Add submodule/code/ folder to sys.path so imports like 'from archs' work
# sys.path.insert(0, str(module_path.parent))  # <-- critical!

# # Load the module
# spec = importlib.util.spec_from_file_location(module_name, module_path)
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)

# # Now the internal import in architectures.py should succeed
# get_architecture = module.get_architecture


# Set random seeds for reproducibility
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(config.seed)
np.random.seed(42)

torch_generator = torch.Generator()
torch_generator.manual_seed(config.seed)

def create_attack(attack_type, dataset_params, attack_params) -> Attack:
    if attack_type == "fgsm":
        return FGSMAttack("FGSM", dataset_params, **attack_params["fgsm"])
    elif attack_type == "rfgsm":
        return RFGSMAttack("RFGSM", dataset_params, **attack_params["rfgsm"])
    elif attack_type == "pgd":
        return PGDAttack("PGD", dataset_params, **attack_params["pgd"])
    elif attack_type == "one_pixel":
        return OnePixelAttack("OnePixel", dataset_params, **attack_params["one_pixel"])
    elif attack_type == "pixle":
        return PixleAttack("Pixle", dataset_params, **attack_params["pixle"])
    elif attack_type == "square":
        return SquareAttack("Square", dataset_params, **attack_params["square"])
    else:
        raise ValueError("Invalid attack type")

def create_attack_dataset(attack, train_loader, validation_loader, models):
    start_time = time.time()
    # Create detaset from fgsm attacked images
    print(f"Creating {attack} attacked dataset...")
    adv_images, org_images, model_idxs = [], [], []
    adv_images_val, org_images_val = [], []
    for i, model in enumerate(models):
        print(f"Running {attack} attack on train dataset of {model.net_type} model...")
        temp_adv_images, temp_org_images = attack.run_attack(model, train_loader)
        print(f"Running {attack} attack on validation dataset of {model.net_type} model...")
        temp_adv_images_val, temp_org_images_val = attack.run_attack(model, validation_loader)
        adv_images.extend(temp_adv_images)
        org_images.extend(temp_org_images)
        adv_images_val.extend(temp_adv_images_val)
        org_images_val.extend(temp_org_images_val)
        model_idxs.extend([i for _ in range(len(temp_adv_images))])

    attack_train_dataset = AttackDataset(adv_images, org_images, model_idxs)
    attack_valid_dataset = AttackDataset(adv_images_val, org_images_val, model_idxs)
    attack_train_loader = DataLoader(attack_train_dataset, batch_size=config.batch_size, shuffle=True, 
                                     worker_init_fn=seed_worker, generator=torch_generator)
    attack_valid_loader = DataLoader(attack_valid_dataset, batch_size=config.batch_size, shuffle=False,
                                     worker_init_fn=seed_worker, generator=torch_generator)

    print("--- Attack dataset created ---")
    print(f"Training {attack} attacked dataset size: {len(attack_train_loader.dataset)}")
    print(f"Validation {attack} attacked dataset size: {len(attack_valid_loader.dataset)}")
    print('--- Total time: %s seconds ---' % (time.time() - start_time))

    return attack_train_loader, attack_valid_loader


if __name__ == "__main__":
    # Set device
    device = set_compute_device()

    print("Testing denoiser configuration:")
    print(f"Dataset: {config.dataset_name}")
    print(f"Attack type: {config.attack_type}")
    print(f"Attack parameters: {config.attack_params}")
    print(f"Batch size: {config.batch_size}")
    print(f"Base classfier models to test: {config.test_model_paths}")
    print(f"Denoiser path: {config.denoiser_path}")
    
    base_dataset = BaseDataset(config.dataset_name, config.batch_size, normalize=False, 
                               torch_generator=torch_generator, sample_percent=config.sample_percent)
    base_dataset.create_dataloaders()
    test_loader = base_dataset.get_test_dataloader()

    print(f"Attack type: {config.attack_type}")
    mean_values, std_values = base_dataset.get_normalization_params()
    dataset_params = {
        "mean": mean_values,
        "std": std_values,
    }
    attack = create_attack(config.attack_type, dataset_params, config.attack_params)

    if config.denoiser_arch == "unet":
        model = load_model(config.denoiser_path, device)
    else:
        checkpoint = torch.load(config.denoiser_path, map_location=device)
        model = get_architecture(checkpoint['arch'], config.dataset_name)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device=device)
    model.eval()

    # Loading classfier models needed for testing
    test_models = []
    for model_path in config.test_model_paths:
        test_models.append(load_model(model_path, device))
        print(f"Loaded model {test_models[-1].net_type}") 
        test_models[-1].eval()

    save_dir = create_save_directories(config.save_root_path)

    total_average_improvement = 0
    # pandas dataframe to save all detailed results
    results = None
    averaged_results = None

    for attacked_model in test_models:
        total_model_improvement = 0
        for attack_params in attack:
            result = None
            result1 = attack.test_attack(attacked_model, test_loader, **attack_params)
            results = pd.concat([results, result1], ignore_index=True) if results is not None else result1
            if config.use_randomized_smoothing:
                attack_params = {**attack_params, "sigma": config.sigma, "n": config.n, "alpha": config.alpha, 
                                 "num_classes": base_dataset.get_num_classes()}
                result2 = attack.test_certified_model(attacked_model, model, test_loader, **attack_params)
            else:
                result2 = attack.test_attack(attacked_model, test_loader, denoiser_model=model, **attack_params)
            results = pd.concat([results, result2], ignore_index=True)
            total_model_improvement += (result2["Accuracy"] - result1["Accuracy"]).iloc[0]

        total_average_improvement += total_model_improvement / len(attack)
        averaged_results_temp = pd.DataFrame({'Model': [attacked_model.net_type], 
                                      'Average Improvement': [total_model_improvement / len(attack)]})
        averaged_results = pd.concat([averaged_results, averaged_results_temp], ignore_index=True) \
                           if averaged_results is not None else averaged_results_temp
        print(f"Average improvement for {attacked_model.net_type}: {100 * total_model_improvement / len(attack)}%")
    print(f"Average improvement for all models: {100 * total_average_improvement / (len(test_models))}%")
    # save results dataframes
    results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    averaged_results.to_csv(os.path.join(save_dir, 'averaged_results.csv'), index=False)