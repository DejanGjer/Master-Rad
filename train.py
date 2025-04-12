import os

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import config_train as config
from resnet18 import ResNet18
from dataset import cifar10_loader_resnet, transform_base_train, transform_base_test
from utils import set_compute_device, create_save_directories, save_config_file

torch.manual_seed(config.seed)
generator = torch.Generator().manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(config.seed)
np.random.seed(config.seed)

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


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    index_store = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            c = pred.eq(target.view_as(pred)).sum().item()
            correct += c
            index_store.append((i, c))
    unload_model(model)

    test_loss /= len(test_loader.dataset)

    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
          .format(model.net_type, test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

    return index_store

def train_normal_model(save_dir, device, train_loader):
    model_normal_name = config.model_info["normal"]["model_path"]
    model_normal_path = os.path.join(save_dir, model_normal_name)
    if os.path.exists(model_normal_path):
        return
    model_normal = ResNet18('normal').to(device)
    optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad,
                                model_normal.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_normal, milestones=config.milestones, gamma=0.1)
    for epoch in range(1, config.epochs + 1):
        train(model_normal, device, train_loader, optimizer_normal,
            epoch, loss_fn=F.cross_entropy)
        scheduler.step()
    model_normal.freeze()
    torch.save(model_normal, model_normal_path)
    unload_model(model_normal)

def train_negative_model(save_dir, device, train_loader):
    model_negative_name = config.model_info["negative"]["model_path"]
    model_negative_path = os.path.join(save_dir, model_negative_name)
    if os.path.exists(model_negative_path):
        return
    model_negative = ResNet18('negative').to(device)
    optimizer_negative = optim.SGD(filter(lambda p: p.requires_grad,
                                model_negative.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_negative, milestones=config.milestones, gamma=0.1)
    for epoch in range(1, config.epochs + 1):
        train(model_negative, device, train_loader, optimizer_negative,
            epoch, loss_fn=F.cross_entropy)
        scheduler.step()
    model_negative.freeze()
    torch.save(model_negative, model_negative_path)
    unload_model(model_negative)

def train_hybrid_nor_model(save_dir, device, train_loader, model_normal):
    model_hybrid_nor_name = config.model_info["hybrid_nor"]["model_path"]
    model_hybrid_nor_path = os.path.join(save_dir, model_hybrid_nor_name)
    if os.path.exists(model_hybrid_nor_path):
        return
    model_hybrid_nor = ResNet18('hybrid_nor').to(device)
    model_hybrid_nor.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    model_hybrid_nor.freeze()
    optimizer_hybrid_nor = optim.SGD(filter(lambda p: p.requires_grad,
                                model_hybrid_nor.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_hybrid_nor, milestones=config.milestones, gamma=0.1)
    for epoch in range(1, config.epochs + 1):
        train(model_hybrid_nor, device, train_loader, optimizer_hybrid_nor,
            epoch, loss_fn=F.cross_entropy)
        scheduler.step()
    torch.save(model_hybrid_nor, model_hybrid_nor_path)
    unload_model(model_hybrid_nor)

def train_hybrid_neg_model(save_dir, device, train_loader, model_negative):
    model_hybrid_neg_name = config.model_info["hybrid_neg"]["model_path"]
    model_hybrid_neg_path = os.path.join(save_dir, model_hybrid_neg_name)
    if os.path.exists(model_hybrid_neg_path):
        return
    model_hybrid_neg = ResNet18('hybrid_neg').to(device)
    model_hybrid_neg.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    model_hybrid_neg.freeze()
    optimizer_hybrid_neg = optim.SGD(filter(lambda p: p.requires_grad,
                                model_hybrid_neg.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_hybrid_neg, milestones=config.milestones, gamma=0.1)
    for epoch in range(1, config.epochs + 1):
        train(model_hybrid_neg, device, train_loader, optimizer_hybrid_neg,
            epoch, loss_fn=F.cross_entropy)
        scheduler.step()
    torch.save(model_hybrid_neg, model_hybrid_neg_path)
    unload_model(model_hybrid_neg)

def get_synergy_nor_model(save_dir, device, model_normal, model_hybrid_nor):
    model_synergy_nor_name = config.model_info["synergy_nor"]["model_path"]
    model_synergy_nor_path = os.path.join(save_dir, model_synergy_nor_name)
    if os.path.exists(model_synergy_nor_path):
        return
    model_synergy_nor = ResNet18('synergy_nor').to(device)
    model_synergy_nor.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    model_synergy_nor.set_linear_normal(model_normal.linear_normal)
    model_synergy_nor.set_linear_normal_n(model_hybrid_nor.linear_normal_n)
    model_synergy_nor.freeze()
    torch.save(model_synergy_nor, model_synergy_nor_path)
    unload_model(model_synergy_nor)

def get_synergy_neg_model(save_dir, device, model_negative, model_hybrid_neg):
    model_synergy_neg_name = config.model_info["synergy_neg"]["model_path"]
    model_synergy_neg_path = os.path.join(save_dir, model_synergy_neg_name)
    if os.path.exists(model_synergy_neg_path):
        return
    model_synergy_neg = ResNet18('synergy_neg').to(device)
    model_synergy_neg.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    model_synergy_neg.set_linear_negative_n(model_negative.linear_negative_n)
    model_synergy_neg.set_linear_negative(model_hybrid_neg.linear_negative)
    model_synergy_neg.freeze()
    torch.save(model_synergy_neg, model_synergy_neg_path)
    unload_model(model_synergy_neg)

def get_synergy_all_model(save_dir, device, model_normal, model_negative, model_hybrid_nor, model_hybrid_neg):
    model_synergy_all_name = config.model_info["synergy_all"]["model_path"]
    model_synergy_all_path = os.path.join(save_dir, model_synergy_all_name)
    if os.path.exists(model_synergy_all_path):
        return
    model_synergy_all = ResNet18('synergy_all').to(device)
    model_synergy_all.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    model_synergy_all.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    model_synergy_all.set_linear_normal(model_normal.linear_normal)
    model_synergy_all.set_linear_normal_n(model_hybrid_nor.linear_normal_n)
    model_synergy_all.set_linear_negative_n(model_negative.linear_negative_n)
    model_synergy_all.set_linear_negative(model_hybrid_neg.linear_negative)
    torch.save(model_synergy_all, model_synergy_all_path)
    unload_model(model_synergy_all)

def train_tr_synergy_all_model(save_dir, device, model_normal, model_negative):
    model_tr_synergy_all_name = config.model_info["tr_synergy_all"]["model_path"]
    model_tr_synergy_all_path = os.path.join(save_dir, model_tr_synergy_all_name)
    if os.path.exists(model_tr_synergy_all_path):
        return
    tr_synergy_all = ResNet18('tr_synergy_all').to(device)
    tr_synergy_all.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    tr_synergy_all.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    tr_synergy_all.freeze()
    optimizer_tr_synergy_all = optim.SGD(filter(lambda p: p.requires_grad,
                                tr_synergy_all.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_tr_synergy_all, milestones=config.milestones, gamma=0.1)
    for epoch in range(1, config.epochs + 1):
        train(tr_synergy_all, device, train_loader, optimizer_tr_synergy_all,
            epoch, loss_fn=F.cross_entropy)
        scheduler.step()
    torch.save(tr_synergy_all, model_tr_synergy_all_path)
    unload_model(tr_synergy_all)

def load_model(model_name, save_dir, device):
    model_path = os.path.join(save_dir, config.model_info[model_name]["model_path"])
    if os.path.exists(model_path):
        return torch.load(model_path, weights_only=False).to(device)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
def unload_model(model):
    del model
    torch.cuda.empty_cache()


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

    save_dir = None
    if config.create_new_saving_dir:
        save_dir = create_save_directories(os.path.join(os.getcwd(), config.checkpoint_dir))
    else:
        save_dir = os.path.join(os.getcwd(), config.checkpoint_dir)
    save_config_file(save_dir, "config_train.py")

    model_normal, model_negative, model_hybrid_nor, model_hybrid_neg = None, None, None, None
    model_synergy_nor, model_synergy_neg, model_synergy_all = None, None, None
    model_tr_synergy_all = None

    model_info = config.model_info

    if model_info["normal"]["train"]:
        print("Training normal model...")
        train_normal_model(save_dir, device, train_loader)
    if model_info["normal"]["test"]:
        print("Testing normal model...")
        model_normal = load_model("normal", save_dir, device)
        test(model_normal, device, test_loader, loss_fn=F.cross_entropy)
        
    if model_info["negative"]["train"]:
        print("Training negative model...")
        train_negative_model(save_dir, device, train_loader)
    if model_info["negative"]["test"]:
        print("Testing negative model...")
        model_negative = load_model("negative", save_dir, device)
        test(model_negative, device, test_loader, loss_fn=F.cross_entropy)

    if model_info["hybrid_nor"]["train"]:
        print("Training hybrid normal model...")
        model_normal = load_model("normal", save_dir, device)
        train_hybrid_nor_model(save_dir, device, train_loader, model_normal)
        unload_model(model_normal)
    if model_info["hybrid_nor"]["test"]:
        print("Testing hybrid normal model...")
        model_hybrid_nor = load_model("hybrid_nor", save_dir, device)
        test(model_hybrid_nor, device, test_loader, loss_fn=F.cross_entropy)

    if model_info["hybrid_neg"]["train"]:
        print("Training hybrid negative model...")
        model_negative = load_model("negative", save_dir, device)
        train_hybrid_neg_model(save_dir, device, train_loader, model_negative)
        unload_model(model_negative)
    if model_info["hybrid_neg"]["test"]:
        print("Testing hybrid negative model...")
        model_hybrid_neg = load_model("hybrid_neg", save_dir, device)
        test(model_hybrid_neg, device, test_loader, loss_fn=F.cross_entropy)

    if model_info["synergy_nor"]["train"]:
        print("Training synergy normal model...")
        model_normal = load_model("normal", save_dir, device)
        model_hybrid_nor = load_model("hybrid_nor", save_dir, device)
        get_synergy_nor_model(save_dir, device, model_normal, model_hybrid_nor)
        unload_model(model_normal)
        unload_model(model_hybrid_nor)
    if model_info["synergy_nor"]["test"]:
        print("Testing synergy normal model...")
        model_synergy_nor = load_model("synergy_nor", save_dir, device)
        test(model_synergy_nor, device, test_loader, loss_fn=F.cross_entropy)

    if model_info["synergy_neg"]["train"]:
        print("Training synergy negative model...")
        model_negative = load_model("negative", save_dir, device)
        model_hybrid_neg = load_model("hybrid_neg", save_dir, device)
        get_synergy_neg_model(save_dir, device, model_negative, model_hybrid_neg)
        unload_model(model_negative)
        unload_model(model_hybrid_neg)
    if model_info["synergy_neg"]["test"]:
        print("Testing synergy negative model...")
        model_synergy_neg = load_model("synergy_neg", save_dir, device)
        test(model_synergy_neg, device, test_loader, loss_fn=F.cross_entropy)

    if model_info["synergy_all"]["train"]:
        print("Training synergy all model...")
        model_normal = load_model("normal", save_dir, device)
        model_negative = load_model("negative", save_dir, device)
        model_hybrid_nor = load_model("hybrid_nor", save_dir, device)
        model_hybrid_neg = load_model("hybrid_neg", save_dir, device)
        get_synergy_all_model(save_dir, device, model_normal, model_negative, model_hybrid_nor, model_hybrid_neg)
        unload_model(model_normal)
        unload_model(model_negative)
        unload_model(model_hybrid_nor)
        unload_model(model_hybrid_neg)
    if model_info["synergy_all"]["test"]:
        print("Testing synergy all model...")
        model_synergy_all = load_model("synergy_all", save_dir, device)
        test(model_synergy_all, device, test_loader, loss_fn=F.cross_entropy)

    if model_info["tr_synergy_all"]["train"]:
        print("Training tr synergy all model...")
        model_normal = load_model("normal", save_dir, device)
        model_negative = load_model("negative", save_dir, device)
        train_tr_synergy_all_model(save_dir, device, model_normal, model_negative)
        unload_model(model_normal)
        unload_model(model_negative)
    if model_info["tr_synergy_all"]["test"]:
        print("Testing tr synergy all model...")
        model_tr_synergy_all = load_model("tr_synergy_all", save_dir, device)
        test(model_tr_synergy_all, device, test_loader, loss_fn=F.cross_entropy)
