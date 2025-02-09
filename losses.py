from torch.nn import L1Loss

def LGDLoss(original, generated, defended_models, model_idxs=None):
    if len(defended_models) == 1:
        defended_model = defended_models[0]
        return L1Loss()(defended_model(original), defended_model(generated))
    else:
        assert model_idxs is not None
        # divide whole batch into sub-batches based on model_idxs
        model_idxs = model_idxs.tolist()
        model_idxs = [int(idx) for idx in model_idxs]
        sub_batches = [[]] * len(defended_models)
        for i, idx in enumerate(model_idxs):
            sub_batches[idx].append(i)
        # calculate loss for each sub-batch
        loss = 0
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
            defended_model = defended_models[i]
            loss += L1Loss()(defended_model(original[sub_batch]), defended_model(generated[sub_batch]))
        return loss / len(model_idxs)