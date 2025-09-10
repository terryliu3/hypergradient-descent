import wandb
from train import train



# # Sweep configuration for comparing different optimizers
# sweep_config_comparison = {
#     'method': 'grid',  # or 'random', 'bayes'
#     'metric': {
#         'name': 'valid/loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'model': {
#             'values': ['logreg', 'mlp']
#         },
#         'method': {
#             'values': ['sgd', 'sgd_hd', 'sgd_hd_kt', 'adam', 'adam_hd', 'adam_hd_kt']
#         },
#         'lr': {
#             'values': [0.1, 0.01, 0.001, 0.0001]
#         },
#         'weight_decay': {
#             'values': [0, 0.0001, 0.001]
#         },
#         'batch_size': {
#             'value': 128
#         },
#         'epochs': {
#             'value': 30
#         },
#         'seed': {
#             'value': 42
#         }
#     }
# }

# # Sweep configuration for SGD-HD hyperparameter tuning
# sweep_config_sgd_hd = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'valid/loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'model': {
#             'value': 'mlp'
#         },
#         'method': {
#             'value': 'sgd_hd'
#         },
#         'lr': {
#             'distribution': 'log_uniform_values',
#             'min': 0.0001,
#             'max': 1.0
#         },
#         'hypergrad_lr': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-8,
#             'max': 1e-3
#         },
#         'weight_decay': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-6,
#             'max': 1e-2
#         },
#         'batch_size': {
#             'values': [64, 128, 256]
#         },
#         'epochs': {
#             'value': 30
#         }
#     }
# }

# Sweep configuration for SGD-HD-KT hyperparameter tuning
sweep_config_mlp_sgd_hd_kt = {
    'method': 'grid',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'value': 'mlp'
        },
        'method': {
            'value': 'sgd_hd_kt'
        },
        'lr': {
            'values': [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        'wealth': {
            'values': [1e-8, 1e-6, 1e-4, 1e-2]
        },
        'epochs': {
            'value': 100
        }
    }
}

sweep_config_mlp_sgd_hd_kt = {
    'method': 'grid',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'value': 'mlp'
        },
        'method': {
            'value': 'sgd_hd_kt'
        },
        'lr': {
            'values': [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        'wealth': {
            'values': [1e-8, 1e-6, 1e-4, 1e-2]
        },
        'epochs': {
            'value': 100
        }
    }
}

sweep_config_mlp_adam_hd_kt = {
    'method': 'grid',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'value': 'mlp'
        },
        'method': {
            'value': 'adam_hd_kt'
        },
        'lr': {
            'values': [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        'wealth': {
            'values': [1e-8, 1e-6, 1e-4, 1e-2]
        },
        'epochs': {
            'value': 100
        }
    }
}
# # Sweep configuration for Adam-HD hyperparameter tuning
# sweep_config_adam_hd = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'valid/loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'model': {
#             'value': 'mlp'
#         },
#         'method': {
#             'value': 'adam_hd'
#         },
#         'lr': {
#             'distribution': 'log_uniform_values',
#             'min': 0.0001,
#             'max': 0.1
#         },
#         'hypergrad_lr': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-10,
#             'max': 1e-5
#         },
#         'beta1': {
#             'distribution': 'uniform',
#             'min': 0.8,
#             'max': 0.99
#         },
#         'beta2': {
#             'distribution': 'uniform',
#             'min': 0.95,
#             'max': 0.9999
#         },
#         'eps': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-10,
#             'max': 1e-6
#         },
#         'weight_decay': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-6,
#             'max': 1e-2
#         },
#         'batch_size': {
#             'values': [64, 128, 256]
#         },
#         'epochs': {
#             'value': 30
#         }
#     }
# }

# Sweep configuration for Adam-HD-KT hyperparameter tuning
# sweep_config_adam_hd_kt = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'valid/loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'model': {
#             'value': 'mlp'
#         },
#         'method': {
#             'value': 'adam_hd_kt'
#         },
#         'lr': {
#             'distribution': 'log_uniform_values',
#             'min': 0.0001,
#             'max': 0.1
#         },
#         'wealth': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-10,
#             'max': 1e-5
#         },
#         'beta1': {
#             'distribution': 'uniform',
#             'min': 0.8,
#             'max': 0.99
#         },
#         'beta2': {
#             'distribution': 'uniform',
#             'min': 0.95,
#             'max': 0.9999
#         },
#         'eps': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-10,
#             'max': 1e-6
#         },
#         'weight_decay': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-6,
#             'max': 1e-2
#         },
#         'batch_size': {
#             'values': [64, 128, 256]
#         },
#         'epochs': {
#             'value': 30
#         }
#     }
# }


# Sweep configuration template for hyperparameter tuning of KT-based optimizers
sweep_config_kt_template = {
    'method': 'grid',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'method': {
            'values': ['sgd_hd_kt', 'sgdn_hd_kt', 'adam_hd_kt']
        },
        'lr': {
            'values': [0, 1e-4, 1e-3, 1e-2]
        },
        'wealth': {
            'values': [1e-7, 1e-5, 1e-3, 1e-1]
        },
    }
}

def run_sweep(model):
    """
    Run a wandb sweep.
    """ 
    sweep_config = sweep_config_kt_template.copy()
    sweep_config['parameters']['model'] = {'value': model}
    if model == 'logreg':
        sweep_config['parameters']['epochs'] = {'value': 10}
    elif model == 'mlp' or model == 'vgg':
        sweep_config['parameters']['epochs'] = {'value': 100}
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="parameter-free-hypergrad")
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train, count=30)  # Adjust count as needed


# def run_custom_sweep():
#     """Run a custom sweep with your own configuration."""
#     custom_config = {
#         'method': 'random',
#         'metric': {
#             'name': 'valid/accuracy',
#             'goal': 'maximize'
#         },
#         'parameters': {
#             'model': {
#                 'values': ['logreg', 'mlp']
#             },
#             'method': {
#                 'values': ['adam_hd', 'adam_hd_kt']
#             },
#             'lr': {
#                 'distribution': 'log_uniform_values',
#                 'min': 0.0001,
#                 'max': 0.01
#             },
#             'wealth': {
#                 'distribution': 'log_uniform_values',
#                 'min': 1e-9,
#                 'max': 1e-5
#             },
#             'hypergrad_lr': {
#                 'distribution': 'log_uniform_values',
#                 'min': 1e-9,
#                 'max': 1e-5
#             },
#             'beta1': {
#                 'value': 0.9
#             },
#             'beta2': {
#                 'value': 0.999
#             },
#             'weight_decay': {
#                 'values': [0, 0.0001]
#             },
#             'batch_size': {
#                 'value': 128
#             },
#             'epochs': {
#                 'value': 20
#             }
#         }
#     }
    
#     sweep_id = wandb.sweep(custom_config, project="hypergrad-optimization")
#     wandb.agent(sweep_id, function=train, count=50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        sweep_type = sys.argv[1]
        run_sweep(sweep_type)
    else:
        print("Usage: python sweep_config.py [comparison|sgd_hd|sgd_hd_kt|adam_hd|adam_hd_kt]")
        print("Running comparison sweep by default...")
        run_sweep('comparison')
