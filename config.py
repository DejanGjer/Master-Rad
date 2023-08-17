# data splits
train_split = 0.8 
validation_split = 0.2 # test set is loaded separately  

# attack parameters
attack_type = 'pgd' # type of the attack can be fgsm and pgd
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
pgd_alpha = 2/255
pgd_steps = 40

# hyperparameters
learning_rate = 0.001
batch_size = 8
epochs = 10
bilinear = True
learn_noise = False
loss = 'pgd' # type of the loss can be pgd and lgd

# model data
model_path = './original_models/model_normal.pt'
save_root_path = './results'