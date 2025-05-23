import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import random

# Blurring function
def add_blur(images, kernel_size=3, sigma=1.0):
    """
    Add Gaussian blur to images
    """
    blurred_images = torch.zeros_like(images)
    for i in range(images.shape[0]):
        blurred_images[i] = F.gaussian_blur(images[i], kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blurred_images

# Random blurring function with varying kernel sizes and sigma
def add_random_blur(images, max_kernel_size=5, max_sigma=2.0):
    """
    Add random Gaussian blur to images
    """
    blurred_images = torch.zeros_like(images)
    for i in range(images.shape[0]):
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.1, max_sigma)
        blurred_images[i] = F.gaussian_blur(images[i], kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blurred_images

# PSNR Calculation
def PSNR(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# Residual Block
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

# Enhanced Generator for Deblurring
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=12):
        super(Generator, self).__init__()
        
        # Initial layers with larger receptive field
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Added layer
            nn.ReLU()
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Increased channels
            nn.ReLU()
        )
        self.down3 = nn.Sequential(  # Added additional downsampling layer
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(512))  # Updated to match new channel size
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(  # Added additional upsampling layer
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)  # Forward through additional downsampling layer
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)  # Forward through additional upsampling layer
        return self.output(x)

# PatchGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Display blurred images
def display_blurred_images(blurred, original, num=5, save_path="blurred_samples.png"):
    fig, axes = plt.subplots(2, num, figsize=(15, 6))
    for i in range(num):
        # Blurred image
        axes[0, i].imshow(blurred[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Blurred')
        
        # Original image
        axes[1, i].imshow(original[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Original')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Blurred samples saved to {save_path}")

# Get a batch of training data
data_iter = iter(train_loader)
images, _ = next(data_iter)

# Add blur and display
blurred_images = add_random_blur(images)
display_blurred_images(blurred_images, images)

# Initialize models
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Determine the output size of the discriminator
dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Example input size (CIFAR-10 images are 32x32)
dummy_output = discriminator(dummy_input)
patch_size = dummy_output.shape[2:]  # Get the spatial dimensions of the output

# Optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss functions
criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()
lambda_l1 = 200  # Weight for L1 loss

# Training loop
epochs = 30
history = {'d_loss': [], 'g_loss': [], 'g_l1_loss': [], 'psnr': []}
best_psnr = 0.0  # Initialize the best PSNR value

print("Starting training...")
for epoch in range(epochs):
    d_loss_total = 0.0
    g_loss_total = 0.0
    g_l1_loss_total = 0.0
    psnr_total = 0.0
    batch_count = 0
    
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Add random blur to create blurred images
        blurred_images = add_random_blur(real_images)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1, *patch_size).to(device)  # Match discriminator output size
        fake_labels = torch.zeros(batch_size, 1, *patch_size).to(device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        discriminator_optimizer.zero_grad()
        
        # Real images
        real_output = discriminator(real_images)
        d_loss_real = criterion_gan(real_output, real_labels)
        
        # Fake images
        generated_images = generator(blurred_images)
        fake_output = discriminator(generated_images.detach())
        d_loss_fake = criterion_gan(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        discriminator_optimizer.step()
        
        # -----------------
        # Train Generator
        # -----------------
        generator_optimizer.zero_grad()
        
        # Generate images and get discriminator output
        generated_images = generator(blurred_images)
        fake_output = discriminator(generated_images)
        
        # GAN loss
        g_loss_gan = criterion_gan(fake_output, real_labels)
        
        # L1 loss (pixel-wise) for sharper images
        g_loss_l1 = criterion_l1(generated_images, real_images)
        
        # Total generator loss
        g_loss = g_loss_gan + lambda_l1 * g_loss_l1
        g_loss.backward()
        generator_optimizer.step()
        
        # Calculate PSNR
        psnr = PSNR(real_images, generated_images)
        
        # Record losses
        d_loss_total += d_loss.item()
        g_loss_total += g_loss_gan.item()
        g_l1_loss_total += g_loss_l1.item()
        psnr_total += psnr.item()
        batch_count += 1
    
    # Calculate average PSNR for the epoch
    avg_psnr = psnr_total / batch_count
    history['d_loss'].append(d_loss_total / batch_count)
    history['g_loss'].append(g_loss_total / batch_count)
    history['g_l1_loss'].append(g_l1_loss_total / batch_count)
    history['psnr'].append(avg_psnr)
    
    # Save the model if the PSNR improves
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(generator.state_dict(), "best_generator_.pth")
        print(f"New best PSNR: {best_psnr:.2f} dB. Model saved.")
    
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"d_loss: {d_loss_total/batch_count:.4f}, "
          f"g_loss: {g_loss_total/batch_count:.4f}, "
          f"g_l1_loss: {g_l1_loss_total/batch_count:.4f}, "
          f"PSNR: {avg_psnr:.2f} dB")

# Test the model
generator.eval()
psnr_values = []

with torch.no_grad():
    for i, (real_images, _) in enumerate(test_loader):
        real_images = real_images.to(device)
        blurred_images = add_random_blur(real_images)
        deblurred_images = generator(blurred_images)
        
        # Calculate PSNR
        psnr = PSNR(real_images, deblurred_images)
        psnr_values.append(psnr.item())
        
        if i == 0:  # Save first batch for visualization
            test_blurred = blurred_images
            test_deblurred = deblurred_images
            test_real = real_images
            break

print(f"Average PSNR on test set: {np.mean(psnr_values):.2f} dB")

# Display and save results
def display_deblurred_images(blurred, deblurred, original, num=5, save_path="deblurring_results.png"):
    fig, axes = plt.subplots(3, num, figsize=(15, 9))
    for i in range(num):
        # Blurred image
        axes[0, i].imshow(blurred[i].cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Blurred')
        
        # Deblurred image
        axes[1, i].imshow(deblurred[i].cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Deblurred')
        
        # Original image
        axes[2, i].imshow(original[i].cpu().permute(1, 2, 0))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Original')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Deblurring results saved to {save_path}")

display_deblurred_images(test_blurred, test_deblurred, test_real)

# Save training history plots
plt.figure(figsize=(12, 6))
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator GAN Loss')
plt.plot(history['g_l1_loss'], label='Generator L1 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig("training_losses_blur.png")
print("Training losses plot saved to 'training_losses.png'")

plt.figure(figsize=(12, 6))
plt.plot(history['psnr'], label='PSNR')
plt.xlabel('Epoch')
plt.ylabel('dB')
plt.legend()
plt.title('Training PSNR')
plt.savefig("training_psnr_blur.png")
print("Training PSNR plot saved to 'training_psnr.png'")