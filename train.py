import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import wandb
from typing import Dict, Any, Optional
from hypergrad import SGDHD, AdamHD, SGDHDKT, AdamHDKT
import vgg

class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

def get_data_loaders(model_name: str, batch_size: int = 128, num_workers: int = 4):
    """Get data loaders based on model type."""
    if model_name in ['logreg', 'mlp']:
        # MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        valid_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
    elif model_name == 'vgg':
        # CIFAR10 dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
        
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                         transform=train_transform, download=True)
        valid_dataset = datasets.CIFAR10(root='./data', train=False,
                                         transform=valid_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return train_loader, valid_loader

def create_model(model_name: str, use_cuda: bool = False, parallel: bool = False):
    """Create model based on name."""
    if model_name == 'logreg':
        model = LogReg(28 * 28, 10)
    elif model_name == 'mlp':
        model = MLP(28 * 28, 1000, 10)
    elif model_name == 'vgg':
        model = vgg.vgg16_bn()
        if parallel:
            model.features = torch.nn.DataParallel(model.features)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if use_cuda:
        model = model.cuda()
    
    return model

def create_optimizer(method: str, model_params, config: Dict[str, Any]):
    """Create optimizer based on method and config."""
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    if method == 'sgd':
        return SGD(model_params, lr=lr, weight_decay=weight_decay)
    
    elif method == 'sgd_hd':
        hypergrad_lr = config.get('hypergrad_lr', 1e-3)
        return SGDHD(model_params, lr=lr, weight_decay=weight_decay, 
                     hypergrad_lr=hypergrad_lr)
    
    elif method == 'sgd_hd_kt':
        wealth = config.get('wealth', 1e-2)
        return SGDHDKT(model_params, lr=lr, weight_decay=weight_decay, 
                       wealth=wealth)
    
    elif method == 'sgdn':
        momentum = config.get('momentum', 0.9)
        return SGD(model_params, lr=lr, weight_decay=weight_decay, 
                   momentum=momentum, nesterov=True)
    
    elif method == 'sgdn_hd':
        momentum = config.get('momentum', 0.9)
        hypergrad_lr = config.get('hypergrad_lr', 1e-6)
        return SGDHD(model_params, lr=lr, weight_decay=weight_decay, 
                     momentum=momentum, nesterov=True, hypergrad_lr=hypergrad_lr)
    
    elif method == 'sgdn_hd_kt':
        momentum = config.get('momentum', 0.9)
        wealth = config.get('wealth', 1e-2)
        return SGDHDKT(model_params, lr=lr, weight_decay=weight_decay,
                       momentum=momentum, nesterov=True, wealth=wealth)
    
    elif method == 'adam':
        return Adam(model_params, lr=lr, weight_decay=weight_decay)
    
    elif method == 'adam_hd':
        hypergrad_lr = config.get('hypergrad_lr', 1e-7)
        return AdamHD(model_params, lr=lr, weight_decay=weight_decay, hypergrad_lr=hypergrad_lr)
    
    elif method == 'adam_hd_kt':
        wealth = config.get('wealth', 1e-2)
        return AdamHDKT(model_params, lr=lr, weight_decay=weight_decay, wealth=wealth)
    else:
        raise ValueError(f"Unknown method: {method}")

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def get_optimizer_stats(optimizer, method: str):
    """Log optimizer-specific statistics to wandb."""
    stats = {}
    group = optimizer.param_groups[0]
    
    # Common stats
    stats['lr'] = group['lr']
    
    # Method-specific stats
    if 'hd' in method:
        if 'hypergrad' in group:
            stats['hypergrad'] = group.get('hypergrad', 0)
        if 'normalized_hypergrad' in group:
            stats['normalized_hypergrad'] = group.get('normalized_hypergrad', 0)
    
    if 'kt' in method:
        if 'wealth' in group:
            stats['wealth'] = group.get('wealth', 0)
        if 'sum_of_normalized_hypergrads' in group:
            stats['sum_of_normalized_hypergrads'] = group.get('sum_of_normalized_hypergrads', 0)
    
    return stats

def train(config: Optional[Dict[str, Any]] = None):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary. If None, uses default values.
    """
    # Default configuration
    default_config = {
        'model': 'logreg',
        'method': 'sgd_hd',
        'lr': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 128,
        'epochs': 10,
        'seed': 1,
        'use_cuda': torch.cuda.is_available(),
        'device': 0,
        'num_workers': 4,
        'parallel': False,
        'log_interval': 1,  # Log every N batches
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-4
    }
    
    # Merge with provided config
    if config is None:
        config = default_config
    
    # Initialize wandb
    wandb.init(project="parameter-free-hypergrad", config=config)
    config = wandb.config
    
    # Set random seeds
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
     
    # Setup device
    device = torch.device(f"cuda:{config.device}" if config.use_cuda else "cpu")
    
    # Create model
    model = create_model(config.model, config.use_cuda, config.parallel)
    wandb.watch(model, log="all", log_freq=100)
    
    # Create data loaders
    train_loader, valid_loader = get_data_loaders(
        config.model, config.batch_size, config.num_workers
    )
    
    # Create optimizer
    optimizer = create_optimizer(config.method, model.parameters(), dict(config))
    
    # Loss function
    criterion = F.cross_entropy
    
    # Training metrics
    best_valid_loss = float('inf')
    patience_counter = 0
    
    # Log initial metrics
    initial_train_loss, initial_train_acc = evaluate(model, train_loader, criterion, device)
    initial_valid_loss, initial_valid_acc = evaluate(model, valid_loader, criterion, device)
    
    wandb.log({
        'epoch': 0,
        'train/loss': initial_train_loss,
        'train/accuracy': initial_train_acc,
        'valid/loss': initial_valid_loss,
        'valid/accuracy': initial_valid_acc,
    })
    
    print(f"Initial - Train Loss: {initial_train_loss:.4f}, Train Acc: {initial_train_acc:.4f}, "
          f"Valid Loss: {initial_valid_loss:.4f}, Valid Acc: {initial_valid_acc:.4f}")
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate batch metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            epoch_train_loss += loss.item() * target.size(0)
            epoch_train_correct += correct
            epoch_train_total += target.size(0)
            
            global_step += 1
            
            # Log batch metrics
            if batch_idx % config.log_interval == 0:
                batch_accuracy = correct / target.size(0)
                optimizer_stats = get_optimizer_stats(optimizer, config.method)
                
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/accuracy': batch_accuracy,
                    'batch/step': global_step,
                    **{f'optimizer/{k}': v for k, v in optimizer_stats.items()}
                })
                
                if batch_idx % (config.log_interval * 10) == 0:
                    print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}, Acc: {batch_accuracy:.4f}, "
                          f"LR: {optimizer_stats['lr']:.6f}")
        
        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / epoch_train_total
        avg_train_acc = epoch_train_correct / epoch_train_total
        
        # Validation
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        
        # Log epoch metrics
        elapsed_time = time.time() - start_time
        optimizer_stats = get_optimizer_stats(optimizer, config.method)
        
        wandb.log({
            'epoch': epoch,
            'train/loss': avg_train_loss,
            'train/accuracy': avg_train_acc,
            'valid/loss': valid_loss,
            'valid/accuracy': valid_acc,
            'time/elapsed': elapsed_time,
            'time/per_epoch': elapsed_time / epoch,
            **{f'optimizer/epoch_{k}': v for k, v in optimizer_stats.items()}
        })
        
        print(f"Epoch {epoch}/{config.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, "
              f"Time: {elapsed_time:.1f}s")
        
        # Early stopping
        if valid_loss < best_valid_loss - config.early_stopping_min_delta:
            best_valid_loss = valid_loss
            patience_counter = 0
            # Save best model
            wandb.run.summary["best_valid_loss"] = valid_loss
            wandb.run.summary["best_valid_accuracy"] = valid_acc
            wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation
    final_train_loss, final_train_acc = evaluate(model, train_loader, criterion, device)
    final_valid_loss, final_valid_acc = evaluate(model, valid_loader, criterion, device)
    
    wandb.run.summary["final_train_loss"] = final_train_loss
    wandb.run.summary["final_train_accuracy"] = final_train_acc
    wandb.run.summary["final_valid_loss"] = final_valid_loss
    wandb.run.summary["final_valid_accuracy"] = final_valid_acc
    wandb.run.summary["total_time"] = time.time() - start_time
    
    print(f"\nTraining completed!")
    print(f"Final - Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.4f}, "
          f"Valid Loss: {final_valid_loss:.4f}, Valid Acc: {final_valid_acc:.4f}")
    
    wandb.finish()
    return model

def main():
    """Main function to run training with custom configuration."""
    config = {
        'model': 'mlp',  # 'logreg', 'mlp', 'vgg'
        'method': 'adam_hd_kt',  # 'sgd', 'sgd_hd', 'sgd_hd_kt', 'adam', 'adam_hd', 'adam_hd_kt', etc.
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 128,
        'epochs': 20,
        'seed': 42,
        
        # Method-specific hyperparameters
        'wealth': 1e-6,  # For KT methods
        'hypergrad_lr': 1e-6,  # For HD methods
        'momentum': 0.9,  # For SGD with momentum
        'beta1': 0.9,  # For Adam
        'beta2': 0.999,  # For Adam
        'eps': 1e-8,  # For Adam
        
        # Training settings
        'log_interval': 10,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-4,
    }
    
    model = train()
    return model


if __name__ == "__main__":
    main()
