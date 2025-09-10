import wandb
import torch
from train import train

models = ['logreg', 'mlp', 'vgg']
methods = ['sgd', 'sgd_hd', 
           'sgdn', 'sgdn_hd',  
           'adam', 'adam_hd']
hypergrad_lrs = [1e-3, 1e-7, 1e-8]
epochs = [10, 100]

config_template = {
    'batch_size': 128,
    'seed': 1,
    'lr': 1e-3,
    
    # System settings
    'use_cuda': torch.cuda.is_available(),
    'device': 0,
    'num_workers': 4,
    'parallel': False,
    
    # Training settings
    # 'log_interval': 1,  # Log every N batches
    # 'early_stopping_patience': 5,
    # 'early_stopping_min_delta': 1e-4,
}

for model in models:
    if model == 'logreg':
        epoch = epochs[0]
    elif model == 'mlp' or model == 'vgg':
        epoch = epochs[1]
    
    for method in methods:
        config = config_template.copy()
        config.update({
                'model': model,
                'method': method,
                'epochs': epoch})

        if 'hd' in method:
            if 'adam' not in method:
                hypergrad_lr = 1e-3
            elif model == 'vgg':
                hypergrad_lr = 1e-8
            else:
                hypergrad_lr = 1e-7
            config['hypergrad_lr'] = hypergrad_lr
        
        train(config)
            
           