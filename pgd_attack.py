from torchattacks import PGD
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from torchvision import transforms
import os

def run_pgd(model, device, loader, epsilons, alpha, steps):
    # loader needs to have batch_size set to 1
    assert len(loader) == len(loader.dataset)
    model.eval()
    adv_images, org_images = [], []
    part_size = len(loader.dataset) // len(epsilons)
    loader = tqdm(loader, total=len(loader))
    print("PGD attack")
    for i, (data, target) in enumerate(loader):
        epsilon = epsilons[i // part_size]
        pgd_attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
        data, target = data.to(device), target.to(device)
        perturbed_data = pgd_attack(data, target)

        adv_images.extend([ex for ex in perturbed_data])
        org_images.extend([ex for ex in data])

    return adv_images, org_images

def test_pgd(model, device, loader, epsilon, alpha, steps, dataset_save_path, batch_size, denoiser=None, dataset_type="test"):
    pgd_attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
    dataset_save_path = os.path.join(dataset_save_path, f"pgd_test_{model.net_type}_{int(epsilon * 100)}.pt")
    pgd_attack.save(data_loader=loader, save_path=dataset_save_path, verbose=False, save_type='int')
    
    adv_dict = torch.load(dataset_save_path)
    adv_images = adv_dict['adv_inputs']
    adv_labels = adv_dict['labels']
    
    transform_att = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    adv_images = transform_att(adv_images.float()/255)     
    adv_data = TensorDataset(adv_images, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0

    for images, labels in adv_loader:
        images, labels = images.to(device), labels.to(device)
        if denoiser is not None:
            images = denoiser(images)
        outputs = model(images)
        final_preds = outputs.max(1, keepdim=False)[1]
        for final, targ in zip(final_preds, labels):
            if final.item() == targ.item():
                correct += 1

    final_acc = correct / float(len(loader.dataset))
    print(
        "Model: {}\t Epsilon: {}\t Dataset: {}\t Denoised: {}\t Accuracy = {} / {} = {}".format(
            model.net_type, epsilon, dataset_type, "Yes" if denoiser else "No", correct, len(loader.dataset), final_acc
        )
    )

    return final_acc

