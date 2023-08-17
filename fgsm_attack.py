import torch.nn.functional as F
from torch.utils.data import Dataset

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # get element-wise signs for gradient ascent
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # clip to [0,1]
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def run_fgsm(model, device, loader, epsilons):
    # loader needs to have batch_size set to 1
    assert len(loader) == len(loader.dataset)
    
    model.eval()
    adv_images, org_images = [], []
    part_size = len(loader.dataset) // len(epsilons)
    for i, (data, target) in enumerate(loader):
        epsilon = epsilons[i // part_size]
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        adv_images.extend([ex for ex in perturbed_data])
        org_images.extend([ex for ex in data])

    return adv_images, org_images

def test_fgsm(model, device, loader, epsilon, save_adversarials=False, denoiser=None, dataset_type="test"):
    model.eval()
    correct = 0
    adv_images, org_images = [], []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        if save_adversarials:
            adv_images.extend([ex for ex in perturbed_data])
            org_images.extend([ex for ex in data])

        init_pred = output.max(1, keepdim=False)[1]
        # If the initial prediction is wrong
        # dont bother attacking, just move on
        for init, targ in zip(init_pred, target):
            if init.item() != targ.item():
                continue
        # if we use denoiser, denoise the perturbed image first
        if denoiser is not None:
            perturbed_data = denoiser(perturbed_data)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=False)[1]
        for final, targ in zip(final_pred, target):
            if final.item() == targ.item():
                correct += 1

    final_acc = correct / float(len(loader.dataset))
    print(
        "Model: {}\t Epsilon: {}\t Dataset: {}\t Denoised: {}\t Accuracy = {} / {} = {}".format(
            model.net_type, epsilon, dataset_type, "Yes" if denoiser else "No", correct, len(loader.dataset), final_acc
        )
    )

    return adv_images, org_images, final_acc