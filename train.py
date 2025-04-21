import os

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm

import config_train as config
from resnet18 import ResNet18
from dataset import cifar10_loader_resnet, transform_base_train, transform_base_test
from utils import set_compute_device, create_save_directories, save_config_file
from dataset import seed_worker, BaseDataset
from train_metrics import TrainMetrics, TestMetrics

torch.manual_seed(config.seed)
generator = torch.Generator().manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(config.seed)
np.random.seed(config.seed)

torch_generator = torch.Generator()
torch_generator.manual_seed(config.seed)

def train(model, device, train_loader, validation_loader, optimizer, epoch, loss_fn, train_metrics=None):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        total_loss += loss.item() * len(data)
        optimizer.step()

    train_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch}, Average Loss: {train_loss:.6f}")
    validation_loss, validation_accuracy = validate(model, device, validation_loader, loss_fn)
    if train_metrics:
        train_metrics.update(train_loss, validation_loss, validation_accuracy)
            
def validate(model, device, validation_loader, loss_fn):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(validation_loader, total=len(validation_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    validation_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print(f"Validation set: Average loss: {validation_loss:.6f}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)")
    return validation_loss, accuracy

def test(model, device, dataset, loss_fn, test_metrics=None):
    test_loader = dataset.get_test_dataloader()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    unload_model(model)
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('[{}] Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'
          .format(model.net_type, test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    if test_metrics:
        test_metrics.update(model.net_type, test_loss, accuracy)


def train_normal_model(save_dir, device, dataset):
    model_normal_name = config.model_info["normal"]["model_path"]
    model_normal_path = os.path.join(save_dir, config.checkpoint_dir, model_normal_name)
    if os.path.exists(model_normal_path):
        print(f"Found existing checkpoint at {model_normal_path}. Skipping training.")
        return
    else:
        os.makedirs(os.path.join(save_dir, config.checkpoint_dir), exist_ok=True)
    train_loader = dataset.get_train_dataloader()
    validation_loader = dataset.get_validation_dataloader()
    model_normal = ResNet18('normal', dataset.num_channels, dataset.num_classes).to(device)
    optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad,
                                model_normal.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_normal, milestones=config.milestones, gamma=0.1)
    train_metrics = TrainMetrics("normal")
    for epoch in range(1, config.epochs + 1):
        train(model_normal, device, train_loader, validation_loader, optimizer_normal,
            epoch, loss_fn=F.cross_entropy, train_metrics=train_metrics)
        scheduler.step()
    model_normal.freeze()
    torch.save(model_normal, model_normal_path)
    train_metrics.save_metrics_to_csv(os.path.join(save_dir, config.metrics_dir))
    train_metrics.plot_metrics(os.path.join(save_dir, config.plot_dir))
    unload_model(model_normal)

def train_negative_model(save_dir, device, dataset):
    model_negative_name = config.model_info["negative"]["model_path"]
    model_negative_path = os.path.join(save_dir, config.checkpoint_dir, model_negative_name)
    if os.path.exists(model_negative_path):
        print(f"Found existing checkpoint at {model_negative_path}. Skipping training.")
        return
    train_loader = dataset.get_train_dataloader()
    validation_loader = dataset.get_validation_dataloader()
    model_negative = ResNet18('negative', dataset.num_channels, dataset.num_classes).to(device)
    optimizer_negative = optim.SGD(filter(lambda p: p.requires_grad,
                                model_negative.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_negative, milestones=config.milestones, gamma=0.1)
    train_metrics = TrainMetrics("negative")
    for epoch in range(1, config.epochs + 1):
        train(model_negative, device, train_loader, validation_loader, optimizer_negative,
            epoch, loss_fn=F.cross_entropy, train_metrics=train_metrics)
        scheduler.step()
    model_negative.freeze()
    torch.save(model_negative, model_negative_path)
    train_metrics.save_metrics_to_csv(os.path.join(save_dir, config.metrics_dir))
    train_metrics.plot_metrics(os.path.join(save_dir, config.plot_dir))
    unload_model(model_negative)

def train_hybrid_nor_model(save_dir, device, dataset, model_normal):
    model_hybrid_nor_name = config.model_info["hybrid_nor"]["model_path"]
    model_hybrid_nor_path = os.path.join(save_dir, config.checkpoint_dir, model_hybrid_nor_name)
    if os.path.exists(model_hybrid_nor_path):
        print(f"Found existing checkpoint at {model_hybrid_nor_path}. Skipping training.")
        return
    train_loader = dataset.get_train_dataloader()
    validation_loader = dataset.get_validation_dataloader()
    model_hybrid_nor = ResNet18('hybrid_nor', dataset.num_channels, dataset.num_classes).to(device)
    model_hybrid_nor.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    model_hybrid_nor.freeze()
    optimizer_hybrid_nor = optim.SGD(filter(lambda p: p.requires_grad,
                                model_hybrid_nor.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_hybrid_nor, milestones=config.milestones, gamma=0.1)
    train_metrics = TrainMetrics("hybrid_nor")
    for epoch in range(1, config.epochs + 1):
        train(model_hybrid_nor, device, train_loader, validation_loader, optimizer_hybrid_nor,
            epoch, loss_fn=F.cross_entropy, train_metrics=train_metrics)
        scheduler.step()
    torch.save(model_hybrid_nor, model_hybrid_nor_path)
    train_metrics.save_metrics_to_csv(os.path.join(save_dir, config.metrics_dir))
    train_metrics.plot_metrics(os.path.join(save_dir, config.plot_dir))
    unload_model(model_hybrid_nor)

def train_hybrid_neg_model(save_dir, device, dataset, model_negative):
    model_hybrid_neg_name = config.model_info["hybrid_neg"]["model_path"]
    model_hybrid_neg_path = os.path.join(save_dir, config.checkpoint_dir, model_hybrid_neg_name)
    if os.path.exists(model_hybrid_neg_path):
        print(f"Found existing checkpoint at {model_hybrid_neg_path}. Skipping training.")
        return
    train_loader = dataset.get_train_dataloader()
    validation_loader = dataset.get_validation_dataloader()
    model_hybrid_neg = ResNet18('hybrid_neg', dataset.num_channels, dataset.num_classes).to(device)
    model_hybrid_neg.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    model_hybrid_neg.freeze()
    optimizer_hybrid_neg = optim.SGD(filter(lambda p: p.requires_grad,
                                model_hybrid_neg.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_hybrid_neg, milestones=config.milestones, gamma=0.1)
    train_metrics = TrainMetrics("hybrid_neg")
    for epoch in range(1, config.epochs + 1):
        train(model_hybrid_neg, device, train_loader, validation_loader, optimizer_hybrid_neg,
            epoch, loss_fn=F.cross_entropy, train_metrics=train_metrics)
        scheduler.step()
    torch.save(model_hybrid_neg, model_hybrid_neg_path)
    train_metrics.save_metrics_to_csv(os.path.join(save_dir, config.metrics_dir))
    train_metrics.plot_metrics(os.path.join(save_dir, config.plot_dir))
    unload_model(model_hybrid_neg)

def get_synergy_nor_model(save_dir, device, dataset, model_normal, model_hybrid_nor):
    model_synergy_nor_name = config.model_info["synergy_nor"]["model_path"]
    model_synergy_nor_path = os.path.join(save_dir, config.checkpoint_dir, model_synergy_nor_name)
    if os.path.exists(model_synergy_nor_path):
        print(f"Found existing checkpoint at {model_synergy_nor_path}. Skipping combining models.")
        return
    model_synergy_nor = ResNet18('synergy_nor', dataset.num_channels, dataset.num_classes).to(device)
    model_synergy_nor.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    model_synergy_nor.set_linear_normal(model_normal.linear_normal)
    model_synergy_nor.set_linear_normal_n(model_hybrid_nor.linear_normal_n)
    model_synergy_nor.freeze()
    torch.save(model_synergy_nor, model_synergy_nor_path)
    unload_model(model_synergy_nor)

def get_synergy_neg_model(save_dir, device, dataset, model_negative, model_hybrid_neg):
    model_synergy_neg_name = config.model_info["synergy_neg"]["model_path"]
    model_synergy_neg_path = os.path.join(save_dir, config.checkpoint_dir, model_synergy_neg_name)
    if os.path.exists(model_synergy_neg_path):
        print(f"Found existing checkpoint at {model_synergy_neg_path}. Skipping combining models.")
        return
    model_synergy_neg = ResNet18('synergy_neg', dataset.num_channels, dataset.num_classes).to(device)
    model_synergy_neg.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    model_synergy_neg.set_linear_negative_n(model_negative.linear_negative_n)
    model_synergy_neg.set_linear_negative(model_hybrid_neg.linear_negative)
    model_synergy_neg.freeze()
    torch.save(model_synergy_neg, model_synergy_neg_path)
    unload_model(model_synergy_neg)

def get_synergy_all_model(save_dir, device, dataset, model_normal, model_negative, model_hybrid_nor, model_hybrid_neg):
    model_synergy_all_name = config.model_info["synergy_all"]["model_path"]
    model_synergy_all_path = os.path.join(save_dir, config.checkpoint_dir, model_synergy_all_name)
    if os.path.exists(model_synergy_all_path):
        print(f"Found existing checkpoint at {model_synergy_all_path}. Skipping combining models.")
        return
    model_synergy_all = ResNet18('synergy_all', dataset.num_channels, dataset.num_classes).to(device)
    model_synergy_all.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    model_synergy_all.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    model_synergy_all.set_linear_normal(model_normal.linear_normal)
    model_synergy_all.set_linear_normal_n(model_hybrid_nor.linear_normal_n)
    model_synergy_all.set_linear_negative_n(model_negative.linear_negative_n)
    model_synergy_all.set_linear_negative(model_hybrid_neg.linear_negative)
    torch.save(model_synergy_all, model_synergy_all_path)
    unload_model(model_synergy_all)

def train_tr_synergy_all_model(save_dir, device, dataset, model_normal, model_negative):
    model_tr_synergy_all_name = config.model_info["tr_synergy_all"]["model_path"]
    model_tr_synergy_all_path = os.path.join(save_dir, config.checkpoint_dir, model_tr_synergy_all_name)
    if os.path.exists(model_tr_synergy_all_path):
        print(f"Found existing checkpoint at {model_tr_synergy_all_path}. Skipping training.")
        return
    train_loader = dataset.get_train_dataloader()
    validation_loader = dataset.get_validation_dataloader()
    tr_synergy_all = ResNet18('tr_synergy_all', dataset.num_channels, dataset.num_classes).to(device)
    tr_synergy_all.set_conv_layers_normal(model_normal.get_conv_layers_normal())
    tr_synergy_all.set_conv_layers_negative(model_negative.get_conv_layers_negative())
    tr_synergy_all.freeze()
    optimizer_tr_synergy_all = optim.SGD(filter(lambda p: p.requires_grad,
                                tr_synergy_all.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_tr_synergy_all, milestones=config.milestones, gamma=0.1)
    train_metrics = TrainMetrics("tr_synergy_all")
    for epoch in range(1, config.epochs + 1):
        train(tr_synergy_all, device, train_loader, validation_loader, optimizer_tr_synergy_all,
            epoch, loss_fn=F.cross_entropy, train_metrics=train_metrics)
        scheduler.step()
    torch.save(tr_synergy_all, model_tr_synergy_all_path)
    train_metrics.save_metrics_to_csv(os.path.join(save_dir, config.metrics_dir))
    train_metrics.plot_metrics(os.path.join(save_dir, config.plot_dir))
    unload_model(tr_synergy_all)

def load_model(model_name, save_dir, device):
    model_path = os.path.join(save_dir, config.checkpoint_dir, config.model_info[model_name]["model_path"])
    if os.path.exists(model_path):
        return torch.load(model_path, weights_only=False).to(device)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
def unload_model(model):
    del model
    torch.cuda.empty_cache()

# def create_datasets():
#     dataset = BaseDataset(config.dataset_name, config.batch_size, config.train_split, 
#                           normalize=False, torch_generator=torch_generator, sample_percent=config.sample_percent)
#     return dataset.get_dataloaders()
    # loader = cifar10_loader_resnet
    # train_loader = loader(device, config.batch_size, transform_base_train, torch_generator=torch_generator, train=True)
    # test_loader = loader(device, config.batch_size, transform_base_test, torch_generator=torch_generator)
    # # split training dataset into training and validation
    # train_dataset, validation_dataset = random_split(train_loader.dataset, 
    #                                                 [config.train_split, config.validation_split],
    #                                                 generator=torch_generator)
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                            batch_size=config.batch_size, 
    #                                            shuffle=True, 
    #                                            worker_init_fn=seed_worker, 
    #                                            generator=torch_generator)
    # validation_loader = torch.utils.data.DataLoader(validation_dataset, 
    #                                                 batch_size=config.batch_size, 
    #                                                 shuffle=False, 
    #                                                 worker_init_fn=seed_worker, 
    #                                                 generator=torch_generator)
    # return train_loader, validation_loader, test_loader


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

    save_dir = None
    if config.create_new_saving_dir:
        save_dir = create_save_directories(os.path.join(os.getcwd(), config.base_save_dir))
    else:
        save_dir = os.path.join(os.getcwd(), config.base_save_dir)
    save_config_file(save_dir, "config_train.py")

    dataset = BaseDataset(config.dataset_name, config.batch_size, config.train_split, 
                          normalize=True, torch_generator=torch_generator, sample_percent=config.sample_percent)
    dataset.create_dataloaders()

    model_info = config.model_info
    test_metrics = TestMetrics(model_info)

    if model_info["normal"]["train"]:
        print("Training normal model...")
        train_normal_model(save_dir, device, dataset)
    if model_info["normal"]["test"]:
        print("Testing normal model...")
        model_normal = load_model("normal", save_dir, device)
        test(model_normal, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)
        
    if model_info["negative"]["train"]:
        print("Training negative model...")
        train_negative_model(save_dir, device, dataset)
    if model_info["negative"]["test"]:
        print("Testing negative model...")
        model_negative = load_model("negative", save_dir, device)
        test(model_negative, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    if model_info["hybrid_nor"]["train"]:
        print("Training hybrid normal model...")
        model_normal = load_model("normal", save_dir, device)
        train_hybrid_nor_model(save_dir, device, dataset, model_normal)
        unload_model(model_normal)
    if model_info["hybrid_nor"]["test"]:
        print("Testing hybrid normal model...")
        model_hybrid_nor = load_model("hybrid_nor", save_dir, device)
        test(model_hybrid_nor, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    if model_info["hybrid_neg"]["train"]:
        print("Training hybrid negative model...")
        model_negative = load_model("negative", save_dir, device)
        train_hybrid_neg_model(save_dir, device, dataset, model_negative)
        unload_model(model_negative)
    if model_info["hybrid_neg"]["test"]:
        print("Testing hybrid negative model...")
        model_hybrid_neg = load_model("hybrid_neg", save_dir, device)
        test(model_hybrid_neg, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    if model_info["synergy_nor"]["train"]:
        print("Training synergy normal model...")
        model_normal = load_model("normal", save_dir, device)
        model_hybrid_nor = load_model("hybrid_nor", save_dir, device)
        get_synergy_nor_model(save_dir, device, dataset, model_normal, model_hybrid_nor)
        unload_model(model_normal)
        unload_model(model_hybrid_nor)
    if model_info["synergy_nor"]["test"]:
        print("Testing synergy normal model...")
        model_synergy_nor = load_model("synergy_nor", save_dir, device)
        test(model_synergy_nor, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    if model_info["synergy_neg"]["train"]:
        print("Training synergy negative model...")
        model_negative = load_model("negative", save_dir, device)
        model_hybrid_neg = load_model("hybrid_neg", save_dir, device)
        get_synergy_neg_model(save_dir, device, dataset, model_negative, model_hybrid_neg)
        unload_model(model_negative)
        unload_model(model_hybrid_neg)
    if model_info["synergy_neg"]["test"]:
        print("Testing synergy negative model...")
        model_synergy_neg = load_model("synergy_neg", save_dir, device)
        test(model_synergy_neg, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    if model_info["synergy_all"]["train"]:
        print("Training synergy all model...")
        model_normal = load_model("normal", save_dir, device)
        model_negative = load_model("negative", save_dir, device)
        model_hybrid_nor = load_model("hybrid_nor", save_dir, device)
        model_hybrid_neg = load_model("hybrid_neg", save_dir, device)
        get_synergy_all_model(save_dir, device, dataset, model_normal, model_negative, model_hybrid_nor, model_hybrid_neg)
        unload_model(model_normal)
        unload_model(model_negative)
        unload_model(model_hybrid_nor)
        unload_model(model_hybrid_neg)
    if model_info["synergy_all"]["test"]:
        print("Testing synergy all model...")
        model_synergy_all = load_model("synergy_all", save_dir, device)
        test(model_synergy_all, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    if model_info["tr_synergy_all"]["train"]:
        print("Training tr synergy all model...")
        model_normal = load_model("normal", save_dir, device)
        model_negative = load_model("negative", save_dir, device)
        train_tr_synergy_all_model(save_dir, device, dataset, model_normal, model_negative)
        unload_model(model_normal)
        unload_model(model_negative)
    if model_info["tr_synergy_all"]["test"]:
        print("Testing tr synergy all model...")
        model_tr_synergy_all = load_model("tr_synergy_all", save_dir, device)
        test(model_tr_synergy_all, device, dataset, loss_fn=F.cross_entropy, test_metrics=test_metrics)

    test_metrics.save_metrics_to_csv(os.path.join(save_dir, config.metrics_dir))
    print("Training and testing completed. Metrics saved.")
