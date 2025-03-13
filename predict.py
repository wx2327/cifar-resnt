import torch
import numpy as np
import pandas as pd
import pickle
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model import se_resnet18

# Set device (MPS for Mac or CUDA for NVIDIA GPU or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Function to load CIFAR-10 data
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# First, check the structure of the Kaggle test file
def inspect_kaggle_test_file(file_path):
    try:
        print(f"Inspecting Kaggle test file: {file_path}")
        test_batch = load_cifar_batch(file_path)
        
        # Print the keys in the file
        print("Keys in the file:")
        for key in test_batch.keys():
            if isinstance(key, bytes):
                print(f"  {key.decode('utf-8')}")
            else:
                print(f"  {key}")
        
        # Check the data structure
        if b'data' in test_batch:
            data = test_batch[b'data']
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            
            # Check the data range
            if data.size > 0:
                print(f"Data range: {data.min()} to {data.max()}")
                if data.max() <= 1.0:
                    print("Note: Data seems to be normalized to [0,1] range")
                elif data.max() <= 255:
                    print("Note: Data seems to be raw pixel values [0,255]")
        
        return test_batch
    except Exception as e:
        print(f"Error inspecting file: {e}")
        return None

# Custom dataset class for handling test data
class CustomCIFAR10TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

# Main function
def main():
    # Model path and parameters
    model_path = 'output/best_se_resnet18.pth'
    width_factor = 0.63
    
    # Kaggle custom test set file path - please replace with the correct path
    kaggle_test_file = 'cifar_test_nolabel.pkl'
    
    submission_output_path = 'submission.csv'
    batch_size = 128
    
    # First, check the structure of the test file
    test_batch = inspect_kaggle_test_file(kaggle_test_file)
    if test_batch is None:
        print("Unable to load test file, exiting")
        return
    
    print("\nLoading model...")
    # Create model and load weights
    model = se_resnet18(num_classes=10, width_factor=width_factor)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    
    print("\nPreparing test data...")
    # Process test data based on inspection results
    data = test_batch[b'data']
    
    # Check if data needs reshaping and transposing
    if len(data.shape) == 2:  # (N, 3072) format
        # Reshape to (N, 3, 32, 32) and then transpose to (N, 32, 32, 3) for image processing
        test_images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        print(f"Reshaped data from {data.shape} to {test_images.shape}")
    else:
        test_images = data
        print(f"Using original data shape: {test_images.shape}")
    
    # Check if data needs normalization
    if test_images.max() > 1.0:
        # If data range is [0,255], convert to [0,1]
        test_images = test_images.astype(np.float32) / 255.0
        print("Normalized data from [0,255] range to [0,1]")
    
    # Test data transformations
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Create test dataset and loader
    test_dataset = CustomCIFAR10TestDataset(test_images, transform=test_transform)
    
    # Avoid using multi-process data loading on Mac
    num_workers = 0 if device.type == 'mps' else 4
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    print(f"Kaggle test dataset loaded with {len(test_dataset)} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # Check prediction distribution
    pred_counts = np.bincount(predictions, minlength=10)
    pred_percentages = pred_counts / len(predictions) * 100
    
    print("\nPrediction class distribution:")
    for i in range(10):
        print(f"Class {i}: {pred_counts[i]} samples ({pred_percentages[i]:.2f}%)")
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': np.arange(len(predictions)), 
        'Labels': predictions
    })
    
    # Save submission file
    submission.to_csv(submission_output_path, index=False)
    print(f"\nSubmission file saved to {submission_output_path}")
    print(f"Total predictions: {len(predictions)}")

if __name__ == "__main__":
    main()