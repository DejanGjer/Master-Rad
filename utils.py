from datetime import datetime as dt
import matplotlib.pyplot as plt
import torch
import os

def load_model(path, device):
    model = None
    if device == torch.device('cuda'):
        model = torch.load(path, weights_only=False)
    else:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    return model

def save_model(model, path):
    torch.save(model, path)

def create_save_directories(root_path, pgd_save_path):
    # create directory if it doesn't exist
    dir_path = os.path.join(root_path, dt.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # create directory for pgd datasets if it doesn't exist
    pgd_dir_path = None
    if pgd_save_path is not None:
        pgd_dir_path = os.path.join(dir_path, pgd_save_path)
        if not os.path.exists(pgd_dir_path):
            os.makedirs(pgd_dir_path)
    return dir_path, pgd_dir_path

def plot_loss_history(metrics, save_path):
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.title("Loss history")
    plt.plot(metrics["train_loss"], label='train')
    plt.plot(metrics["validation_loss"], label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_history.png'))

def set_compute_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device == torch.device('cpu'):
        print("WARNING: Dataset is going to be sampled!")
    return device

