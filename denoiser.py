from tqdm import tqdm
from torch import nn
import torch

from losses import LGDLoss

def train_denoiser(model, train_loader, optimizer, loss_type, device, defended_models):
    loss_fn = nn.MSELoss()
    if loss_type == "lgd":
        loss_fn = LGDLoss

    model.train()
    loss_history = []
    # setup tqdm
    train_loader = tqdm(train_loader, total=len(train_loader))
    # Train
    for data, target, model_idx in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target, defended_models, model_idx)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        
    return loss_history

def test_denoiser(model, test_loader, loss_type, device, defended_models):
    loss_fn = nn.MSELoss()
    if loss_type == "lgd":
        loss_fn = LGDLoss

    model.eval()
    with torch.no_grad():
        loss_history = []
        # setup tqdm
        test_loader = tqdm(test_loader, total=len(test_loader))
        for data, target, model_idx in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target, defended_models, model_idx)
            loss_history.append(loss.item())
    return loss_history