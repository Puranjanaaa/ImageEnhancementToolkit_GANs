import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import random

# Define the same Generator architecture as in training
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=12):
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(512))
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.output(x)

# Define the same blurring functions as in training
def add_random_blur(images, max_kernel_size=5, max_sigma=2.0):
    blurred_images = torch.zeros_like(images)
    for i in range(images.shape[0]):
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.7, max_sigma)
        blurred_images[i] = F.gaussian_blur(images[i], kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blurred_images

# Initialize the generator
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# Load the trained weights
generator.load_state_dict(torch.load("models/best_generator_deblur.pth", map_location=device))
generator.eval()

# Load CIFAR-10 test set with the same transform as training
transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)  # Get 5 random images

# Get a batch of test images
test_images, _ = next(iter(test_loader))

# Create blurred versions (using the same function as in training)
blurred_images = add_random_blur(test_images)

# Move to device
test_images = test_images.to(device)
blurred_images = blurred_images.to(device)

# Generate deblurred images
with torch.no_grad():
    deblurred_images = generator(blurred_images)

# Convert tensors to numpy arrays for visualization
def tensor_to_numpy(tensor):
    return tensor.cpu().numpy().transpose(0, 2, 3, 1)

blurred_np = tensor_to_numpy(blurred_images)
deblurred_np = tensor_to_numpy(deblurred_images)
original_np = tensor_to_numpy(test_images)

# Create output directory if it doesn't exist
os.makedirs("deblurring_results", exist_ok=True)

# Display and save comparison results
plt.figure(figsize=(15, 9))
for i in range(5):
    # Blurred image
    plt.subplot(3, 5, i+1)
    plt.imshow(blurred_np[i])
    plt.axis('off')
    if i == 0:
        plt.title('Blurred')
    
    # Deblurred image
    plt.subplot(3, 5, i+6)
    plt.imshow(deblurred_np[i])
    plt.axis('off')
    if i == 0:
        plt.title('Deblurred')
    
    # Original image
    plt.subplot(3, 5, i+11)
    plt.imshow(original_np[i])
    plt.axis('off')
    if i == 0:
        plt.title('Original')

plt.tight_layout()
plt.savefig("deblurring_results/deblur_comparison.png")
plt.show()

# Save individual images
for i in range(5):
    # Save blurred
    plt.imsave(f"deblurring_results/blurred_{i}.png", blurred_np[i])
    
    # Save deblurred
    plt.imsave(f"deblurring_results/deblurred_{i}.png", deblurred_np[i])
    
    # Save original
    plt.imsave(f"deblurring_results/original_{i}.png", original_np[i])

print("Results saved in 'deblurring_results' directory")