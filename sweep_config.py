import wandb
from train import train


# Sweep configuration for comparing different optimizers
sweep_config_comparison = {
    'method': 'grid',  # or 'random', 'bayes'
    'metric': {
        'name': 'valid/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'values': ['logreg', 'mlp']
        },
        'method': {
            'values': ['sgd', 'sgd_hd', 'sgd_hd_kt', 'adam', 'adam_hd', 'adam_hd_kt']
        },
        'lr': {
            'values': [0.1, 0.01, 0.001, 0.0001]
        },
        'weight_decay': {
            'values': [0, 0.0001, 0.001]
        },
        'batch_size': {
            'value': 128
        },
        'epochs': {
            'value': 30
        },
        'seed': {
            'value': 42
        }
    }
}

# Sweep configuration for SGD-HD hyperparameter tuning
sweep_config_sgd_hd = {
    'method': 'bayes',
    'metric': {
        'name': 'valid/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'value': 'mlp'
        },
        'method': {
            'value': 'sgd_hd'
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1.0
        },
        'hypergrad_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-8,
            'max': 1e-3
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'epochs': {
            'value': 30
        }
    }
}

# Sweep configuration for SGD-HD-KT hyperparameter tuning
sweep_config_sgd_hd_kt = {
    'method': 'bayes',
    'metric': {
        'name': 'valid/loss',
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
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1.0
        },
        'wealth': {
            'distribution': 'log_uniform_values',
            'min': 1e-8,
            'max': 1e-3
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'epochs': {
            'value': 30
        }
    }
}

# Sweep configuration for Adam-HD hyperparameter tuning
sweep_config_adam_hd = {
    'method': 'bayes',
    'metric': {
        'name': 'valid/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'value': 'mlp'
        },
        'method': {
            'value': 'adam_hd'
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'hypergrad_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-10,
            'max': 1e-5
        },
        'beta1': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'beta2': {
            'distribution': 'uniform',
            'min': 0.95,
            'max': 0.9999
        },
        'eps': {
            'distribution': 'log_uniform_values',
            'min': 1e-10,
            'max': 1e-6
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'epochs': {
            'value': 30
        }
    }
}

# Sweep configuration for Adam-HD-KT hyperparameter tuning
sweep_config_adam_hd_kt = {
    'method': 'bayes',
    'metric': {
        'name': 'valid/loss',
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
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'wealth': {
            'distribution': 'log_uniform_values',
            'min': 1e-10,
            'max': 1e-5
        },
        'beta1': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'beta2': {
            'distribution': 'uniform',
            'min': 0.95,
            'max': 0.9999
        },
        'eps': {
            'distribution': 'log_uniform_values',
            'min': 1e-10,
            'max': 1e-6
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'epochs': {
            'value': 30
        }
    }
}


def run_sweep(sweep_type='comparison'):
    """
    Run a wandb sweep.
    
    Args:
        sweep_type: Type of sweep to run. Options:
            - 'comparison': Compare all optimizers
            - 'sgd_hd': Tune SGD-HD hyperparameters
            - 'sgd_hd_kt': Tune SGD-HD-KT hyperparameters
            - 'adam_hd': Tune Adam-HD hyperparameters
            - 'adam_hd_kt': Tune Adam-HD-KT hyperparameters
    """
    # Select sweep configuration
    configs = {
        'comparison': sweep_config_comparison,
        'sgd_hd': sweep_config_sgd_hd,
        'sgd_hd_kt': sweep_config_sgd_hd_kt,
        'adam_hd': sweep_config_adam_hd,
        'adam_hd_kt': sweep_config_adam_hd_kt
    }
    
    if sweep_type not in configs:
        raise ValueError(f"Unknown sweep type: {sweep_type}. Choose from {list(configs.keys())}")
    
    sweep_config = configs[sweep_type]
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="hypergrad-optimization")
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train, count=100)  # Adjust count as needed


def run_custom_sweep():
    """Run a custom sweep with your own configuration."""
    custom_config = {
        'method': 'random',
        'metric': {
            'name': 'valid/accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'model': {
                'values': ['logreg', 'mlp']
            },
            'method': {
                'values': ['adam_hd', 'adam_hd_kt']
            },
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.01
            },
            'wealth': {
                'distribution': 'log_uniform_values',
                'min': 1e-9,
                'max': 1e-5
            },
            'hypergrad_lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-9,
                'max': 1e-5
            },
            'beta1': {
                'value': 0.9
            },
            'beta2': {
                'value': 0.999
            },
            'weight_decay': {
                'values': [0, 0.0001]
            },
            'batch_size': {
                'value': 128
            },
            'epochs': {
                'value': 20
            }
        }
    }
    
    sweep_id = wandb.sweep(custom_config, project="hypergrad-optimization")
    wandb.agent(sweep_id, function=train, count=50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        sweep_type = sys.argv[1]
        run_sweep(sweep_type)
    else:
        print("Usage: python sweep_config.py [comparison|sgd_hd|sgd_hd_kt|adam_hd|adam_hd_kt]")
        print("Running comparison sweep by default...")
        run_sweep('comparison')
