from torchattacks import PGD
from torch.utils.data import Dataset

def run_pgd(model, device, loader, epsilons, alpha, steps):
    # loader needs to have batch_size set to 1
    assert len(loader) == len(loader.dataset)
    model.eval()
    adv_images, org_images = [], []
    part_size = len(loader.dataset) // len(epsilons)
    for i, (data, target) in enumerate(loader):
        epsilon = epsilons[i // part_size]
        pgd_attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
        data, target = data.to(device), target.to(device)
        perturbed_data = pgd_attack(data, target)

        adv_images.extend([ex for ex in perturbed_data])
        org_images.extend([ex for ex in data])

    return adv_images, org_images