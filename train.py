import time

import torch
import torch.optim as optim
import torch.nn.functional as F

import config_train as config
from resnet18 import ResNet18
from dataset import cifar10_loader_resnet, transform_base_train, transform_base_test
from utils import set_compute_device


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5000 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(model.net_type, epoch, batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_fn, dataset_name=None, filter_map=None):
    model.eval()
    test_loss = 0
    correct = 0
    index_store = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if filter_map and dataset_name in filter_map:
              for batch in data:
                for channel in batch:
                  channel.mul_(filter_map[dataset_name].to(device))
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            c = pred.eq(target.view_as(pred)).sum().item()
            correct += c
            index_store.append((i, c))

    test_loss /= len(test_loader.dataset)

    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
          .format(model.net_type, test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

    return index_store

def print_training_info(device):
    print("Configuration details:")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Momentum: {config.momentum}")
    print(f"Weight decay: {config.decay}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Milestones: {config.milestones}")
    print("---------------------------------")
    print(f"Device used for training: {device}")


if __name__ == "__main__":
    # Set device
    device = set_compute_device()

    print_training_info(device)

    loader = cifar10_loader_resnet

    train_loader = loader(device, config.batch_size, transform_base_train, train=True)
    test_loader = loader(device, config.batch_size, transform_base_test)

    model_normal = ResNet18('normal').to(device)

    # ---- Normal net:

    optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad,
                                model_normal.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_normal, milestones=config.milestones, gamma=0.1)
    start_time = time.time()


    for epoch in range(1, config.epochs + 1):
        train(model_normal, device, train_loader, optimizer_normal,
            epoch, loss_fn=F.cross_entropy)
        scheduler.step()

    model_normal.freeze()
    torch.save(model_normal, 'model_normal.pt')

