checkpoint_dir = 'checkpoints'
create_new_saving_dir = True

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
        "train": True,
        "test": True
    },
    "hybrid_neg": {
        "model_path": "model_hybrid_neg.pth",
        "train": True,
        "test": True
    },
    "synergy_nor": {
        "model_path": "model_synergy_nor.pth",
        "train": True,
        "test": True
    },
    "synergy_neg": {
        "model_path": "model_synergy_neg.pth",
        "train": True,
        "test": True
    },
    "synergy_all": {
        "model_path": "model_synergy_all.pth",
        "train": True,
        "test": True
    },
    "tr_synergy_all": {
        "model_path": "model_tr_synergy_all.pth",
        "train": True,
        "test": True
    }
}

learning_rate = 0.1
momentum = 0.9
decay = 5e-3
epochs = 50
batch_size = 768
milestones = [30, 40]