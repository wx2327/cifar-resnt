import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# Squeeze and Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# SE-ResNet Block (ResNet block with Squeeze and Excitation)
class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# Modified ResNet architecture for CIFAR-10
class ModifiedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, width_factor=1.0):
        super(ModifiedResNet, self).__init__()
        self.in_channels = int(64 * width_factor)
        
        # Initial convolutional layer - reduced kernel size for CIFAR-10
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Removed maxpool for CIFAR-10's small images
        
        # Residual layers
        self.layer1 = self._make_layer(block, int(64 * width_factor), layers[0])
        self.layer2 = self._make_layer(block, int(128 * width_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_factor), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_factor), layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_factor), num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Function to create ResNet18
def resnet18(num_classes=10, width_factor=1.0):
    return ModifiedResNet(ResidualBlock, [2, 2, 2, 2], num_classes, width_factor)

# Function to create ResNet34
def resnet34(num_classes=10, width_factor=1.0):
    return ModifiedResNet(ResidualBlock, [3, 4, 6, 3], num_classes, width_factor)

# Function to create SE-ResNet18 (with Squeeze and Excitation blocks)
def se_resnet18(num_classes=10, width_factor=1.0):
    return ModifiedResNet(SEResidualBlock, [2, 2, 2, 2], num_classes, width_factor)

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test the model architecture and parameter count
if __name__ == "__main__":
    # Create model instances with different width factors
    models = {
        "ResNet18 (w=0.5)": resnet18(width_factor=0.5),
        "ResNet18 (w=0.7)": resnet18(width_factor=0.7),
        "ResNet18 (w=1.0)": resnet18(width_factor=1.0),
        "SE-ResNet18 (w=0.5)": se_resnet18(width_factor=0.5),
        "SE-ResNet18 (w=0.7)": se_resnet18(width_factor=0.7),
        "ResNet34 (w=0.5)": resnet34(width_factor=0.5),
    }
    
    # Check parameter counts
    print("Model Parameter Counts:")
    print("-" * 50)
    for name, model in models.items():
        num_params = count_parameters(model)
        print(f"{name}: {num_params:,} parameters")
        
    # Test forward pass with a sample input
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create a sample input
    x = torch.randn(1, 3, 32, 32).to(device)
    
    # Test SE-ResNet18 forward pass
    model = se_resnet18(width_factor=0.7).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")