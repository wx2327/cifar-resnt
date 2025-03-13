# CIFAR-10 Image Classification

## Project Description
This repository contains code for a CIFAR-10 image classification model using a modified SE-ResNet18 architecture. The project is part of a Deep Learning course assignment at NYU Tandon School of Engineering.

## Author
- Name: Wenxuan Wu
- Email: wx2327@nyu.edu

## Repository Structure
- `model.py`: Contains the model architecture definition (SE-ResNet18 with width factor)
- `train.py`: Script for training the model
- `predict.py`: Script for making predictions on the test set

## Requirements
- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm

## How to Run

### Training
To train the model:
```
python train.py
```
This will:
1. Download and prepare the CIFAR-10 dataset
2. Train an SE-ResNet18 model with width factor 0.63
3. Save the best model to `output/best_se_resnet18.pth`
4. Generate training curves

### Prediction
To generate predictions for submission:
```
python predict.py
```
This will:
1. Load the trained model
2. Make predictions on the provided test set
3. Save predictions to `submission.csv`

## Model Architecture
The model uses a Squeeze-and-Excitation ResNet18 architecture modified for CIFAR-10, with:
- Width factor of 0.63 to reduce parameters
- Initial 3x3 convolution instead of 7x7 to preserve resolution
- No initial max-pooling layer
- Approximately 4.45 million parameters

## Performance
The model achieves:
- 96.00% validation accuracy on the CIFAR-10 validation set
- 92.51% accuracy on the standard CIFAR-10 test set
- 80.91% accuracy on the Kaggle custom test set
