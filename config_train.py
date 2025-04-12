checkpoint_dir = 'checkpoints'
create_new_saving_dir = True
models_to_train = {
    "normal": True,
    "negative": True,
    "hybrid_nor": True,
    "hybrid_neg": True,
    "synergy_nor": True,
    "synergy_neg": True,
    "synergy_all": True,
    "tr_synergy_all": True
}
model_name_to_path = {
    "normal": "model_normal.pth",
    "negative": "model_negative.pth",
    "hybrid_nor": "model_hybrid_nor.pth",
    "hybrid_neg": "model_hybrid_neg.pth",
    "synergy_nor": "model_synergy_nor.pth",
    "synergy_neg": "model_synergy_neg.pth",
    "synergy_all": "model_synergy_all.pth",
    "tr_synergy_all": "model_tr_synergy_all.pth"
}

learning_rate = 0.1
momentum = 0.9
decay = 5e-3
epochs = 50
batch_size = 768
milestones = [30, 40]