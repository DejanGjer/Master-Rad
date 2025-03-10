from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchattacks import PGD, FGSM, OnePixel
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

from dataset import CIFAR10_MEAN, CIFAR10_STD
from utils import normalize_images

class Attack(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_attack(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the attack on the model using the given dataloader"""
        pass

    @abstractmethod
    def test_attack(self, model: nn.Module, dataloader: DataLoader, denoser_model: nn.Module = None, **kwargs) -> pd.Series:
        """Test the attack on the model using the given dataloader and denoiser model if available"""
        pass

    def metrics(self, correct: int, model: nn.Module, dataloader: DataLoader, denoiser_model: nn.Module = None, **kwargs) -> pd.DataFrame:
        """Calculate test metrics and return as a pandas Series"""
        final_acc = correct / float(len(dataloader.dataset))
        # create one row in the results dataframe
        result = pd.DataFrame({
            'Model': [model.net_type], 
            'Denoised': ["Yes"] if denoiser_model else ["No"]
        })
        result = pd.concat([result, pd.DataFrame(kwargs)], axis=1)
        metrics = pd.DataFrame({
            'Correct': [correct],
            'Total': [len(dataloader.dataset)],
            'Accuracy': [final_acc]
        })
        result = pd.concat([result, metrics], axis=1)
        print(result)
        # print the results
        table = [result.index.tolist(), result.values.tolist()]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
        return result

    def __str__(self) -> str:
        return self.name

class FGSMAttack(Attack):
    def __init__(self, name, epsilons):
        super().__init__(name)
        self.epsilons = epsilons

    def run_attack(self, model, dataloader):
        fgsm_attacks = [FGSM(model, eps=epsilon) for epsilon in self.epsilons]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.epsilons)
        assert part_size > 0, "Too many epsilons for the dataset size"
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.epsilons) else i % len(self.epsilons)
            perturbed_data = fgsm_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            data = normalize_images(data, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            adv_images.extend(perturbed_data)
            org_images.extend(data)

        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, epsilon=0.01):
        fgsm_attack = FGSM(model, eps=epsilon)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with FGSM attack with epsilon = {epsilon}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = fgsm_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, epsilon=epsilon)

class PGDAttack(Attack):
    def __init__(self, name, epsilons, alpha, steps):
        super().__init__(name)
        self.epsilons = epsilons
        self.alpha = alpha
        self.steps = steps

    def run_attack(self, model, dataloader):
        pgd_attacks = [PGD(model, eps=epsilon, alpha=self.alpha, steps=self.steps) for epsilon in self.epsilons]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.epsilons)
        assert part_size > 0, "Too many epsilons for the dataset size"
        print("PGD attack")
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.epsilons) else i % len(self.epsilons) 
            perturbed_data = pgd_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            data = normalize_images(data, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, epsilon=0.01):
        pgd_attack = PGD(model, eps=epsilon, alpha=self.alpha, steps=self.steps)
        # dataset_save_path = os.path.join(dataset_save_path, f"pgd_test_{model.net_type}_{int(epsilon * 100)}.pt")
        # pgd_attack.save(data_loader=dataloader, save_path=dataset_save_path, verbose=False, save_type='int')
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with PGD attack with epsilon = {epsilon}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = pgd_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, epsilon=epsilon)

class OnePixelAttack(Attack):
    def __init__(self, name, pixel_counts, steps, popsize, batch_size):
        super().__init__(name)
        self.pixel_count = pixel_counts
        self.steps = steps
        self.popsize = popsize
        self.batch_size = batch_size

    def run_attack(self, model, dataloader):
        one_pixel_attacks = [OnePixel(model, pixel_count=pc, steps=self.steps, popsize=self.popsize, inf_batch=self.batch_size) for pc in self.pixel_count]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.pixel_count)
        assert part_size > 0, "Too many pixel counts for the dataset size"
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.pixel_count) else i % len(self.pixel_count)
            perturbed_data = one_pixel_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            data = normalize_images(data, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, pixel_count=1):
        one_pixel_attack = OnePixel(model, pixel_count=pixel_count, steps=self.steps, popsize=self.popsize, inf_batch=self.batch_size)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with OnePixel attack with pixel count = {pixel_count}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = one_pixel_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, pixel_count=pixel_count)