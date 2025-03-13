import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import time
from model import resnet18, resnet34, se_resnet18, count_parameters

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# Cutout implementation
class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
        
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
            
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

# Data augmentation and loading
def get_data_loaders(batch_size=128, num_workers=4, use_cutout=True):
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    if use_cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))
    
    # No augmentation for validation/test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Split training data for validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# Mixup augmentation
def mixup_data(x, y, alpha=1.0, device='mps'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixup augmentation
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * inputs.size(0)
        
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate epoch metrics
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total if not use_mixup else float('nan')
    
    return train_loss, train_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track metrics
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate validation metrics
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

# Main training loop
def main():
    # Training settings
    model_type = 'se_resnet18'  # Options: 'resnet18', 'resnet34', 'se_resnet18'
    width_factor = 0.63  # 
    batch_size = 256
    epochs = 50
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    use_mixup = True
    mixup_alpha = 1.0
    use_cutout = True
    seed = 42
    output_dir = './output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Set device (use MPS for Mac)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size, use_cutout=use_cutout)
    print(f"Dataset sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # Create model
    if model_type == 'resnet18':
        model = resnet18(num_classes=10, width_factor=width_factor)
    elif model_type == 'resnet34':
        model = resnet34(num_classes=10, width_factor=width_factor)
    elif model_type == 'se_resnet18':
        model = se_resnet18(num_classes=10, width_factor=width_factor)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model: {model_type} with width factor {width_factor}")
    print(f"Number of parameters: {num_params:,}")
    
    # Check if model is within parameter limit
    param_limit = 5_000_000
    if num_params > param_limit:
        print(f"WARNING: Model exceeds parameter limit of {param_limit:,}")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize tracking variables
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup, mixup_alpha)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if not np.isnan(train_acc):
            train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%" if not np.isnan(train_acc) else f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_{model_type}.pth"))
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Report training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_{model_type}.pth")))
    
    # Evaluate on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    if train_accs:
        plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
    plt.plot(range(1, epochs+1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_training_curves.png"))
    plt.show()
    
    # Save final results
    results = {
        "model_type": model_type,
        "width_factor": width_factor,
        "num_params": num_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "training_time_minutes": total_time/60
    }
    
    print("\nFinal Results Summary:")
    print(f"Model: {results['model_type']} with width factor {results['width_factor']}")
    print(f"Number of parameters: {results['num_params']:,}")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Test accuracy: {results['test_acc']:.2f}%")
    print(f"Training time: {results['training_time_minutes']:.2f} minutes")

if __name__ == "__main__":
    main()