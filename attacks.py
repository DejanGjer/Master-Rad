from abc import ABC, abstractmethod
from typing import List, Tuple
import inspect
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchattacks import PGD, FGSM, RFGSM, OnePixel, Pixle, Square
from tqdm import tqdm
import pandas as pd
import math

from utils import normalize_images

class Attack(ABC):
    def __init__(self, name: str, dataset_params: dict):
        self.name = name
        self.dataset_params = dataset_params

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
        result = pd.concat([result, pd.DataFrame(kwargs, index=[0])], axis=1)
        metrics = pd.DataFrame({
            'Correct': [correct],
            'Total': [len(dataloader.dataset)],
            'Accuracy': [final_acc]
        })
        result = pd.concat([result, metrics], axis=1)
        print(result)
        return result
    
    def __iter__(self):
        # Find all class attributes that are lists
        list_fields = {
            name: value
            for name, value in inspect.getmembers(self)
            if isinstance(value, list)
        }
        if not list_fields:
            # If tehre are no list fields, return iterartor with one dummy element
            yield {}
        else:
            # Yield all combinations of list fields
            for combination in itertools.product(*list_fields.values()):
                yield dict(zip(list_fields.keys(), combination))

    def __len__(self):
        # Find all class attributes that are lists
        list_fields = {
            name: value
            for name, value in inspect.getmembers(self)
            if isinstance(value, list)
        }
        # Return the product of the lengths of all list fields
        return math.prod(len(v) for v in list_fields.values()) if list_fields else 1

    def __str__(self) -> str:
        return self.name

class FGSMAttack(Attack):
    def __init__(self, name, dataset_params, epsilons):
        super().__init__(name, dataset_params)
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
            perturbed_data = normalize_images(perturbed_data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            data = normalize_images(data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, epsilons=0.01):
        # rename the epsilons argument (which has to have same name as the class attribute) to epsilon
        epsilon=epsilons
        fgsm_attack = FGSM(model, eps=epsilon)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with FGSM attack with epsilon = {epsilon}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = fgsm_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, epsilon=epsilon)
    
class RFGSMAttack(Attack):
    def __init__(self, name, dataset_params, epsilons, alpha, steps):
        super().__init__(name, dataset_params)
        self.epsilons = epsilons
        self.alpha = alpha
        self.steps = steps

    def run_attack(self, model, dataloader):
        rfgsm_attacks = [RFGSM(model, eps=epsilon, alpha=self.alpha, steps=self.steps) for epsilon in self.epsilons]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.epsilons)
        assert part_size > 0, "Too many epsilons for the dataset size"
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.epsilons) else i % len(self.epsilons)
            perturbed_data = rfgsm_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            data = normalize_images(data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, epsilons=0.01):
        # rename the epsilons argument (which has to have same name as the class attribute) to epsilon
        epsilon=epsilons
        rfgsm_attack = RFGSM(model, eps=epsilon, alpha=self.alpha, steps=self.steps)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with RFGSM attack with epsilon = {epsilon}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = rfgsm_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, epsilon=epsilon)

class PGDAttack(Attack):
    def __init__(self, name, dataset_params, epsilons, alpha, steps):
        super().__init__(name, dataset_params)
        self.epsilons = epsilons
        self.alpha = alpha
        self.steps = steps

    def run_attack(self, model, dataloader):
        pgd_attacks = [PGD(model, eps=epsilon, alpha=self.alpha, steps=self.steps) for epsilon in self.epsilons]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.epsilons)
        assert part_size > 0, "Too many epsilons for the dataset size"
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.epsilons) else i % len(self.epsilons) 
            perturbed_data = pgd_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            data = normalize_images(data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, epsilons=0.01):
        # rename the epsilons argument (which has to have same name as the class attribute) to epsilon
        epsilon=epsilons
        pgd_attack = PGD(model, eps=epsilon, alpha=self.alpha, steps=self.steps)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with PGD attack with epsilon = {epsilon}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = pgd_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, epsilon=epsilon)
    
class OnePixelAttack(Attack):
    def __init__(self, name, dataset_params, pixel_counts, steps, popsize, batch_size):
        super().__init__(name, dataset_params)
        self.pixel_counts = pixel_counts
        self.steps = steps
        self.popsize = popsize
        self.batch_size = batch_size

    def run_attack(self, model, dataloader):
        one_pixel_attacks = [OnePixel(model, pixels=pc, steps=self.steps, popsize=self.popsize, inf_batch=self.batch_size) for pc in self.pixel_counts]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.pixel_counts)
        assert part_size > 0, "Too many pixel counts for the dataset size"
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.pixel_counts) else i % len(self.pixel_counts)
            perturbed_data = one_pixel_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            data = normalize_images(data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, pixel_counts=1):
        # rename the pixel_counts argument (which has to have same name as the class attribute) to pixel_count
        pixel_count = pixel_counts
        one_pixel_attack = OnePixel(model, pixels=pixel_count, steps=self.steps, popsize=self.popsize, inf_batch=self.batch_size)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with OnePixel attack with pixel count = {pixel_count}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = one_pixel_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, pixel_count=pixel_count)
    
class PixleAttack(Attack):
    def __init__(self, name, dataset_params, x_dimensions, y_dimensions, pixel_mapping, 
                 restarts, max_iterations, update_each_iteration):
        super().__init__(name, dataset_params)
        self.x_dimensions = x_dimensions
        self.y_dimensions = y_dimensions
        self.pixel_mapping = pixel_mapping
        self.restarts = restarts
        self.max_iterations = max_iterations
        self.update_each_iteration = update_each_iteration

    def run_attack(self, model, dataloader):
        pixle_attacks = [Pixle(model, x_dimensions=self.x_dimensions, y_dimensions=self.y_dimensions, 
                               pixel_mapping=self.pixel_mapping, restarts=self.restarts, max_iterations=self.max_iterations, 
                               update_each_iteration=self.update_each_iteration)]
        model.eval()
        adv_images, org_images = [], []
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            perturbed_data = pixle_attacks[0](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            data = normalize_images(data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None):
        pixle_attack = Pixle(model, x_dimensions=self.x_dimensions, y_dimensions=self.y_dimensions, 
                             pixel_mapping=self.pixel_mapping, restarts=self.restarts, max_iterations=self.max_iterations, 
                             update_each_iteration=self.update_each_iteration)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with Pixle attack, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = pixle_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model)
    
class SquareAttack(Attack):
    def __init__(self, name, dataset_params, norm, epsilons, n_queries, n_restarts, p_init, loss, resc_schedule, seed):
        super().__init__(name, dataset_params)
        self.norm = norm
        self.epsilons = epsilons
        self.n_queries = n_queries
        self.n_restarts = n_restarts
        self.p_init = p_init
        self.loss = loss
        self.resc_schedule = resc_schedule
        self.seed = seed

    def run_attack(self, model, dataloader):
        square_attacks = [Square(model, norm=self.norm, eps=epsilon, n_queries=self.n_queries, n_restarts=self.n_restarts, 
                                 p_init=self.p_init, loss=self.loss, resc_schedule=self.resc_schedule, seed=self.seed) for epsilon in self.epsilons]
        model.eval()
        adv_images, org_images = [], []
        part_size = len(dataloader) // len(self.epsilons)
        assert part_size > 0, "Too many epsilons for the dataset size"
        for i, (data, target) in enumerate(tqdm(dataloader, total=len(dataloader))):
            part = i // part_size if i // part_size < len(self.epsilons) else i % len(self.epsilons)
            perturbed_data = square_attacks[part](data, target)
            perturbed_data = normalize_images(perturbed_data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            data = normalize_images(data, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            adv_images.extend(perturbed_data)
            org_images.extend(data)
        return adv_images, org_images
    
    def test_attack(self, model, dataloader, denoiser_model=None, epsilons=0.01):
        # rename the epsilons argument (which has to have same name as the class attribute) to epsilon
        epsilon=epsilons
        square_attack = Square(model, norm=self.norm, eps=epsilon, n_queries=self.n_queries, n_restarts=self.n_restarts, 
                               p_init=self.p_init, loss=self.loss, resc_schedule=self.resc_schedule, seed=self.seed)
        model.eval()
        correct = 0
        print(f"Testing {model.net_type} with Square attack with epsilon = {epsilon}, denoised = {denoiser_model is not None}")
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            adv_images = square_attack(images, labels)
            adv_images = normalize_images(adv_images, mean=self.dataset_params["mean"], std=self.dataset_params["std"])
            if denoiser_model is not None:
                adv_images = denoiser_model(adv_images)
            outputs = model(adv_images)
            final_preds = outputs.max(1, keepdim=False)[1]
            for final, targ in zip(final_preds, labels):
                if final.item() == targ.item():
                    correct += 1
        return self.metrics(correct, model, dataloader, denoiser_model=denoiser_model, epsilon=epsilon)

