from abc import ABC, abstractmethod
from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchattacks import PGD, FGSM
from tqdm import tqdm
import pandas as pd

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
        final_acc = correct / float(len(dataloader.dataset))
        # create one row in the results dataframe
        result = pd.Series({'Model': model.net_type, 
                            'Epsilon': epsilon,  
                            'Denoised': "Yes" if denoiser_model else "No", 
                            'Accuracy': final_acc})
        print(
            "Model: {}\t Epsilon: {}\t Denoised: {}\t Accuracy = {} / {} = {}".format(
                model.net_type, epsilon, "Yes" if denoiser_model else "No", correct, len(dataloader.dataset), final_acc
            )
        )
        return result

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

        final_acc = correct / float(len(dataloader.dataset))
        # create one row in the results dataframe
        result = pd.Series({'Model': model.net_type, 
                            'Epsilon': epsilon,  
                            'Denoised': "Yes" if denoiser_model else "No", 
                            'Accuracy': final_acc})
        print(
            "Model: {}\t Epsilon: {}\t Denoised: {}\t Accuracy = {} / {} = {}".format(
                model.net_type, epsilon, "Yes" if denoiser_model else "No", correct, len(dataloader.dataset), final_acc
            )
        )
        return result
