# dataset
dataset_name = 'cifar100' # cifar10, cifar100, mnist, imagenette
sample_percent = None # None for all data, 0.1 for 10% of the data, etc.

# attack parameters
attack_type = 'fgsm'
attack_params = {
    "fgsm": {
        "epsilons": [0.01, 0.02, 0.03, 0.04, 0.05]
    },
    "rfgsm": {
        "epsilons": [0.01, 0.02, 0.03, 0.04, 0.05],
        "alpha": 2/255,
        "steps": 40
    },
    "pgd": {
        "epsilons": [0.01, 0.02, 0.03, 0.04, 0.05],
        "alpha": 2/255,
        "steps": 40
    },
    "one_pixel": {
        "pixel_counts": [1],
        "steps": 10,
        "popsize": 10,
        "batch_size": 128
    },
    "pixle": {
        "x_dimensions": (2,10),
        "y_dimensions": (2,10),
        "pixel_mapping": "random",
        "restarts": 20,
        "max_iterations": 10,
        "update_each_iteration": False
    },
    "square": {
        "norm": "Linf", # # L2 or Linf
        "epsilons": [0.01, 0.02, 0.03, 0.04, 0.05],
        "n_queries": 5000,
        "n_restarts": 1,
        "p_init": 0.8,
        "loss": "ce", # margin or ce
        "resc_schedule": True,
        "seed": 42,
    },
}

# hyperparameters
batch_size = 128

# model paths
test_model_paths = [
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_normal.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_negative.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_hybrid_nor.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_hybrid_neg.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_synergy_nor.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_synergy_neg.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_synergy_all.pth',
    './base_training_checkpoints/2025-04-21_09-03-11/checkpoints/model_tr_synergy_all.pth'
]

denoiser_path = './denoiser_checkpoints/2025-04-21_09-03-11/checkpoints/denoiser.pth'
denoiser_type = 'unet'  # 'unet' or one of available arch from denoised-smoothing submodule in architectures.py DENOISERS_ARCHITECTURES

# saving paths
save_root_path = './results'
