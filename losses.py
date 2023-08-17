from torch.nn import L1Loss

def LGDLoss(original, generated, defended_model):
    return L1Loss()(defended_model(original), defended_model(generated))