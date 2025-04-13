checkpoint_dir = 'checkpoints/2025-04-13_12-07-20'
create_new_saving_dir = False
seed = 42

model_info = {
    "normal": {
        "model_path": "model_normal.pth",
        "train": True,
        "test": True
    },
    "negative": {
        "model_path": "model_negative.pth",
        "train": True,
        "test": True
    },
    "hybrid_nor": {
        "model_path": "model_hybrid_nor.pth",
        "train": False,
        "test": False
    },
    "hybrid_neg": {
        "model_path": "model_hybrid_neg.pth",
        "train": False,
        "test": False
    },
    "synergy_nor": {
        "model_path": "model_synergy_nor.pth",
        "train": False,
        "test": False
    },
    "synergy_neg": {
        "model_path": "model_synergy_neg.pth",
        "train": False,
        "test": False
    },
    "synergy_all": {
        "model_path": "model_synergy_all.pth",
        "train": False,
        "test": False
    },
    "tr_synergy_all": {
        "model_path": "model_tr_synergy_all.pth",
        "train": False,
        "test": False
    }
}

# data splits
train_split = 0.8 
validation_split = 0.2 # test set is loaded separately 

learning_rate = 0.1
momentum = 0.9
decay = 5e-3
epochs = 50
batch_size = 768
milestones = [30, 40]