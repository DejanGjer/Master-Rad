from torchattacks import PGD
from tqdm import tqdm
import os
import pandas as pd

def run_pgd(model, loader, epsilons, alpha, steps):
    pgd_attacks = [PGD(model, eps=epsilon, alpha=alpha, steps=steps) for epsilon in epsilons]
    model.eval()
    adv_images, org_images = [], []
    part_size = len(loader) // len(epsilons)
    assert part_size > 0, "Too many epsilons for the dataset size"
    print("PGD attack")
    for i, (data, target) in enumerate(tqdm(loader, total=len(loader))):
        part = i // part_size if i // part_size < len(epsilons) else i % len(epsilons) 
        perturbed_data = pgd_attacks[part](data, target)
        adv_images.extend(perturbed_data)
        org_images.extend(data)

    return adv_images, org_images

def test_pgd(model, loader, epsilon, alpha, steps, dataset_save_path, denoiser=None, dataset_type="test"):
    pgd_attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
    dataset_save_path = os.path.join(dataset_save_path, f"pgd_test_{model.net_type}_{int(epsilon * 100)}.pt")
    pgd_attack.save(data_loader=loader, save_path=dataset_save_path, verbose=False, save_type='int')

    model.eval()
    correct = 0
    print(f"Testing {model.net_type} on {dataset_type} dataset with PGD attack with epsilon = {epsilon}, denoised = {denoiser is not None}")
    for images, labels in tqdm(loader, total=len(loader)):
        adv_images = pgd_attack(images, labels)
        if denoiser is not None:
            adv_images = denoiser(adv_images)
        outputs = model(adv_images)
        final_preds = outputs.max(1, keepdim=False)[1]
        for final, targ in zip(final_preds, labels):
            if final.item() == targ.item():
                correct += 1

    final_acc = correct / float(len(loader.dataset))
    # create one row in the results dataframe
    result = pd.Series({'Model': model.net_type, 
                        'Epsilon': epsilon, 
                        'Dataset': dataset_type, 
                        'Denoised': "Yes" if denoiser else "No", 
                        'Accuracy': final_acc})
    print(
        "Model: {}\t Epsilon: {}\t Dataset: {}\t Denoised: {}\t Accuracy = {} / {} = {}".format(
            model.net_type, epsilon, dataset_type, "Yes" if denoiser else "No", correct, len(loader.dataset), final_acc
        )
    )

    return final_acc, result

