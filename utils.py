from datetime import datetime as dt
from typing import List

import matplotlib.pyplot as plt
import torch
import os
import shutil

def normalize_images(images: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    assert torch.max(images) <= 1.0 and torch.min(images) >= 0.0, "Images are not in range [0, 1]"
    assert len(mean) == len(std), f"Mean and std must have same length, got {len(mean)} and {len(std)}"
    device = images.device
    num_channels = len(mean)  # Get number of channels from the mean list
    assert images.shape[1] == num_channels, f"Expected {num_channels} channels but got {images.shape[1]}"
    mean = torch.tensor(mean).view(1, num_channels, 1, 1).to(device)
    std = torch.tensor(std).view(1, num_channels, 1, 1).to(device)
    return (images - mean) / std

def load_model(path, device):
    model = None
    if device == torch.device('cuda'):
        model = torch.load(path, weights_only=False)
    else:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    return model

def save_model(model, path):
    torch.save(model, path)

def has_timestamped_directory(path: str) -> bool:
    # checks if some directory on a path already contains a timestamp format
    for dirname in os.path.normpath(os.path.abspath(path)).split(os.sep):
        # check if directory name ,astches this time format %Y-%m-%d_%H-%M-%S
        try:
            if dt.strptime(dirname, "%Y-%m-%d_%H-%M-%S"):
                print(f"Found timestamped directory: {dirname}")
                return True
        except ValueError:
            continue
    return False

def create_save_directories(root_path):
    dir_path = root_path
    # Check if the root path already has a timestamped directory if not add it
    if not has_timestamped_directory(root_path):
        dir_path = os.path.join(root_path, dt.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # Create the directory if it does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def save_config_file(save_dir: str, filename: str):
    config_save_path = os.path.join(save_dir, filename)
    shutil.copyfile(filename, config_save_path)

def plot_loss_history(metrics, save_path):
    # Plot loss history over epochs
    plt.figure(figsize=(10, 5))
    plt.title("Loss history")
    plt.plot(metrics["train_loss"], label='train')
    plt.plot(metrics["validation_loss"], label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)

def plot_accuracy_history(metrics, save_path):
    # Plot accuracy history over epochs
    plt.figure(figsize=(10, 5))
    plt.title("Accuracy history")
    plt.plot(metrics, label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_path)

def set_compute_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device == torch.device('cpu'):
        print("WARNING: Dataset is runnigng on cpu and is possibly automatically sampled!")
    return device

