# data splits
train_split = 0.8 
validation_split = 0.2 # test set is loaded separately  

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
learning_rate = 0.001
batch_size = 128
epochs = 15
bilinear = True
learn_noise = True
loss = 'lgd' # type of the loss can be pgd and lgd
seed = 42

# model paths
train_model_paths = [
    './original_models/model_normal.pt',
]
test_model_paths = [
    './original_models/model_normal.pt',
    './original_models/model_negative.pt',
    './original_models/model_hybrid_nor.pt',
    './original_models/model_hybrid_neg.pt',
    './original_models/model_synergy_nor.pt',
    './original_models/model_synergy_neg.pt',
    './original_models/model_synergy_all.pt',
    './original_models/model_tr_synergy_all.pt'
]

# saving paths
save_root_path = './results'
pgd_save_path = 'pgd_datasets'
