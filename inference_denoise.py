import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

# Define the same Generator architecture
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
    def __init__(self, num_residual_blocks=6):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(64))
        self.mid_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.additional_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.new_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.residual_conv = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        initial_out = self.initial(x)
        res_out = self.res_blocks(initial_out)
        mid_out = self.mid_layer(res_out)
        additional_out = self.additional_layer(mid_out)
        new_out = self.new_layer(additional_out)
        residual = self.residual_conv(x)
        out = new_out + residual
        out = self.relu(out)
        return self.output(out)

# Salt and pepper noise function (same as training)
def add_salt_and_pepper_noise_rgb(images, prob=0.1):
    noisy_images = images.clone()
    batch_size, channels, height, width = noisy_images.shape
    
    # Salt noise
    num_salt = int(prob * height * width * 0.5)
    salt_coords = [torch.randint(0, dim, (batch_size, num_salt)) for dim in (height, width)]
    for i in range(batch_size):
        noisy_images[i, :, salt_coords[0][i], salt_coords[1][i]] = 1.0
    
    # Pepper noise
    num_pepper = int(prob * height * width * 0.5)
    pepper_coords = [torch.randint(0, dim, (batch_size, num_pepper)) for dim in (height, width)]
    for i in range(batch_size):
        noisy_images[i, :, pepper_coords[0][i], pepper_coords[1][i]] = 0.0
    
    return noisy_images

# PSNR Calculation (same as training)
def PSNR(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# Initialize and load the trained generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("models/best_generator_denoised_3.pth", map_location=device))
generator.eval()

# Load CIFAR-10 test set with same transform
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)  # Small batch for visualization

# Get a batch of test images
test_images, _ = next(iter(test_loader))

# Add noise (same parameters as final test in training)
noisy_images = add_salt_and_pepper_noise_rgb(test_images, prob=0.05).to(device)

# Denoise the images
with torch.no_grad():
    denoised_images = generator(noisy_images)

# Convert to numpy for visualization
def tensor_to_numpy(tensor):
    return tensor.cpu().numpy().transpose(0, 2, 3, 1)

noisy_np = tensor_to_numpy(noisy_images)
denoised_np = tensor_to_numpy(denoised_images)
original_np = tensor_to_numpy(test_images)

# Calculate PSNR
psnr_values = []
for i in range(len(test_images)):
    psnr = PSNR(test_images[i].to(device), denoised_images[i])
    psnr_values.append(psnr.item())
avg_psnr = np.mean(psnr_values)

print(f"Average PSNR: {avg_psnr:.2f} dB")

# Display and save results
def display_results(noisy, denoised, original, num=5, save_path="denoising_results.png"):
    plt.figure(figsize=(15, 9))
    for i in range(num):
        # Noisy
        plt.subplot(3, num, i+1)
        plt.imshow(noisy[i])
        plt.axis('off')
        if i == 0:
            plt.title('Noisy')
        
        # Denoised
        plt.subplot(3, num, i+num+1)
        plt.imshow(denoised[i])
        plt.axis('off')
        if i == 0:
            plt.title('Denoised')
        
        # Original
        plt.subplot(3, num, i+2*num+1)
        plt.imshow(original[i])
        plt.axis('off')
        if i == 0:
            plt.title('Original')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")

display_results(noisy_np, denoised_np, original_np)

# Optionally: Process the entire test set and save PSNR statistics
all_psnr = []
with torch.no_grad():
    for batch, _ in test_loader:
        noisy_batch = add_salt_and_pepper_noise_rgb(batch, prob=0.4).to(device)
        denoised_batch = generator(noisy_batch)
        psnr = PSNR(batch.to(device), denoised_batch)
        all_psnr.append(psnr.item())

print(f"Final test set PSNR: {np.mean(all_psnr):.2f} dB (σ={np.std(all_psnr):.2f})")