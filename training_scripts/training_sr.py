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
from torchvision.models import vgg19
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from math import log10

# Enhanced downsampling with more realistic degradation
def downsample_images(images, scale_factor=2):
    """Create low-resolution images with realistic degradation pipeline"""
    # Add slight Gaussian blur - reduced sigma for less blurring
    blurred = F.gaussian_blur(images, kernel_size=[3, 3], sigma=[0.3, 1.2])
    
    # Downsample with bicubic interpolation
    _, _, h, w = images.shape
    low_res = F.resize(blurred, size=[h//scale_factor, w//scale_factor], 
                     interpolation=transforms.InterpolationMode.BICUBIC)
    
    # Add less compression artifacts
    low_res = torch.clamp(low_res + torch.randn_like(low_res)*0.01, -1, 1)
    
    # Upsample back to original size
    return F.resize(low_res, size=[h, w], 
                  interpolation=transforms.InterpolationMode.BICUBIC)

# Enhanced Residual Channel Attention Block (RCAB)
# Suppresses less useful features while preserving critical information
# Residual connection preserves original information
class RCAB(nn.Module):
    def __init__(self, channels, reduction=8):  # Increased reduction for better channel attention
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),  # Removed bias
            nn.LeakyReLU(0.2, inplace=True),  # Changed to LeakyReLU
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),  # Added BatchNorm
            ChannelAttention(channels, reduction)
        )
    
    def forward(self, x):
        return x + self.body(x)

# Focuses on "what" features are important by recalibrating channel-wise feature responses using 
# both average and max pooling information.
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling for better feature extraction
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        y = self.sigmoid(avg_out + max_out)  # Combines both avg and max features
        return x * y

# Spatial Attention Module for added feature enhancement
# Focuses on "where" features are important in the spatial dimensions by generating a 2D attention map.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

# Enhanced RCAB with spatial attention
# Combines channel attention (what) with spatial attention (where) to create a comprehensive dual attention mechanism.
class EnhancedRCAB(nn.Module):
    def __init__(self, channels, reduction=8):
        super(EnhancedRCAB, self).__init__()
        self.channel_attention = RCAB(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Residual Group containing multiple enhanced RCABs
# Stacks multiple EnhancedRCAB blocks together with a global residual connection to enable very deep feature extraction.
class ResidualGroup(nn.Module):
    def __init__(self, n_blocks, channels):
        super(ResidualGroup, self).__init__()
        blocks = [EnhancedRCAB(channels) for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        self.blocks = nn.Sequential(*blocks)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.blocks(x))  # Added activation after skip connection

# Dense connection helper
class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels + channels//2, channels, kernel_size=3, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        out1 = self.lrelu(self.conv1(x))
        out = torch.cat([x, out1], 1)
        out = self.conv2(out)
        out = self.bn(out)
        return self.lrelu(out)

# Enhanced Generator with Residual-in-Residual Dense Blocks and CBAM
class GeneratorSR(nn.Module):
    def __init__(self, scale_factor=2, n_groups=10, n_blocks=20, channels=128):  # Increased base channels
        super(GeneratorSR, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Deep feature extraction with residual groups
        self.body = nn.ModuleList([
            ResidualGroup(n_blocks, channels) for _ in range(n_groups)
        ])
        
        # Dense connections between groups
        self.fusion = nn.Conv2d(n_groups * channels, channels, 1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
        # Global skip connection
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
        # Enhanced upsampling with ICNR initialization
        upsampling = []
        for _ in range(int(np.log2(scale_factor))):
            upsampling.extend([
                nn.Conv2d(channels, channels*4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                DenseBlock(channels)  # Added dense block after each upsampling
            ])
        self.upsample = nn.Sequential(*upsampling)
        
        # Output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Apply ICNR initialization to PixelShuffle layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        
        # Process through residual groups with dense connections
        features = []
        x_temp = x1
        for i, block in enumerate(self.body):
            x_temp = block(x_temp)
            features.append(x_temp)
        
        # Fuse multi-level features
        x = self.fusion(torch.cat(features, dim=1))
        x = self.bn(x)
        x = self.lrelu(x)
        
        # Global skip connection
        x = self.conv2(x) + x1
        
        # Upsampling and final output
        x = self.upsample(x)
        return self.conv3(x)

# Enhanced Perceptual Loss with VGG19 features from multiple layers, 
# Pixel-based losses (MSE/L1) produce overly smooth images
# Pixel losses can't distinguish between perceptually important differences
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features.eval()
        self.vgg_slices = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
            vgg[18:27] # relu4_4
        ])
        
        for param in self.vgg_slices.parameters():
            param.requires_grad = False
            
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
        self.criterion = nn.L1Loss()
        
    def forward(self, sr, hr):
        # Normalize to ImageNet stats for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(sr.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(sr.device)
        
        sr = (sr + 1) / 2  # [-1,1] to [0,1]
        hr = (hr + 1) / 2
        
        sr = (sr - mean) / std
        hr = (hr - mean) / std
        
        loss = 0
        for i, block in enumerate(self.vgg_slices):
            sr = block(sr)
            hr = block(hr)
            loss += self.weights[i] * self.criterion(sr, hr)
            
        return loss
    
# Enhanced Discriminator with Spectral Normalization
class DiscriminatorSR(nn.Module):
    def __init__(self):
        super(DiscriminatorSR, self).__init__()
        
        def spectral_norm(module):
            return nn.utils.spectral_norm(module)
        
        self.model = nn.Sequential(
            # Block 1
            spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 5
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 6
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 7
            spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 8
            spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Adaptive pooling and final layers
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(512, 1024, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(1024, 1, kernel_size=1))
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        return self.model(x).view(batch_size, -1)

# Enhanced PSNR Calculation with clipping
def PSNR(y_true, y_pred):
    y_true = torch.clamp(y_true, -1, 1)
    y_pred = torch.clamp(y_pred, -1, 1)
    
    # Convert from [-1,1] to [0,1]
    y_true = (y_true + 1) / 2
    y_pred = (y_pred + 1) / 2
    
    mse = torch.mean((y_true - y_pred) ** 2)
    return 10 * torch.log10(1.0 / mse)

# Load the saved model and continue training

# Training Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 dataset with improved augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Slight rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize models
generator = GeneratorSR(scale_factor=2, n_groups=8, n_blocks=16, channels=128).to(device)
discriminator = DiscriminatorSR().to(device)

# Load the saved generator model
print("Loading saved model: best_generator_sr.pth")
generator.load_state_dict(torch.load("best_generator_sr_9.pth"))

# Loss functions
perceptual_loss = PerceptualLoss().to(device) #Perceptual loss computes the difference between high-level features extracted from a pretrained network (typically VGG19)
l1_loss = nn.L1Loss() #L1 loss computes the mean absolute difference between the predicted image and the ground truth.
bce_loss = nn.BCEWithLogitsLoss() #It measures the probability that an input image is real or generated (fake)
# 
# Optimizers with improved settings
g_optim = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)  # Reduced learning rate
d_optim = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-5)  # Reduced learning rate

# Learning rate schedulers
g_scheduler = CosineAnnealingLR(g_optim, T_max=50, eta_min=1e-7)
d_scheduler = CosineAnnealingLR(d_optim, T_max=50, eta_min=1e-7)

# Loss weights - adjusted for better balance
lambda_pixel = 10.0
lambda_perceptual = 1.0
lambda_gan = 0.1

# Training loop with validation
best_psnr = 0.0
history = {'d_loss': [], 'g_loss': [], 'psnr': [], 'val_psnr': []}

# Add gradient clipping threshold
grad_clip = 1.0

# Start from epoch 10 (since we already trained for 10 epochs)
start_epoch = 30

# First, evaluate the loaded model to see what PSNR we're starting with
generator.eval()
val_psnr = 0.0
val_batches = 0

with torch.no_grad():
    for hr_images, _ in test_loader:
        hr_images = hr_images.to(device)
        lr_images = downsample_images(hr_images)
        _, _, h, w = hr_images.shape
        hr_images_upscaled = F.resize(hr_images, size=[h*2, w*2], 
                                    interpolation=transforms.InterpolationMode.BICUBIC)
        sr_images = generator(lr_images)
        val_psnr += PSNR(hr_images_upscaled, sr_images).item()
        val_batches += 1

initial_psnr = val_psnr / val_batches
best_psnr = initial_psnr  # Set this as our starting best PSNR
print(f"Starting PSNR from loaded model: {initial_psnr:.2f} dB")

# Continue training for 10 more epochs
for epoch in range(start_epoch, start_epoch + 10):
    generator.train()
    discriminator.train()

    d_loss_total = 0.0
    g_loss_total = 0.0
    psnr_total = 0.0
    batch_count = 0
    
    for i, (hr_images, _) in enumerate(train_loader):
        batch_size = hr_images.size(0)
        hr_images = hr_images.to(device)
        
        # Apply random crop augmentation
        _, _, h, w = hr_images.shape
        hr_images_upscaled = F.resize(hr_images, size=[h*2, w*2], 
                                    interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Generate low-res images with improved degradation model
        lr_images = downsample_images(hr_images)
        
        # ===== Train Discriminator =====
        d_optim.zero_grad()
        
        # Real images
        real_pred = discriminator(hr_images_upscaled)
        d_loss_real = bce_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake images
        sr_images = generator(lr_images).detach()
        fake_pred = discriminator(sr_images)
        d_loss_fake = bce_loss(fake_pred, torch.zeros_like(fake_pred))
        
        # Add gradient penalty for WGAN-GP style training
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolated = alpha * hr_images_upscaled + (1 - alpha) * sr_images
        interpolated.requires_grad_(True)
        
        interpolated_pred = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
        
        d_loss = d_loss_real + d_loss_fake + gradient_penalty
        d_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
        
        d_optim.step()
        
        # ===== Train Generator =====
        if i % 1 == 0:  # Train generator every iteration
            g_optim.zero_grad()
            
            sr_images = generator(lr_images)
            
            # Pixel-wise loss - L1
            pixel_loss = l1_loss(sr_images, hr_images_upscaled)
            
            # Perceptual loss
            percep_loss = perceptual_loss(sr_images, hr_images_upscaled)
            
            # GAN loss with label smoothing
            fake_pred = discriminator(sr_images)
            g_loss_gan = bce_loss(fake_pred, torch.ones_like(fake_pred) * 0.9)  # Label smoothing
            
            # Total loss with adjusted weights
            g_loss = (lambda_pixel * pixel_loss + 
                     lambda_perceptual * percep_loss + 
                     lambda_gan * g_loss_gan)
            
            g_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            
            g_optim.step()
        
        # Calculate PSNR
        with torch.no_grad():
            psnr = PSNR(hr_images_upscaled, sr_images)

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()
        psnr_total += psnr.item()
        batch_count += 1
        
        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {i}: "
                  f"G Loss: {g_loss.item():.4f}, "
                  f"D Loss: {d_loss.item():.4f}, "
                  f"PSNR: {psnr.item():.2f} dB")
    
    # Calculate average metrics for the epoch
    avg_psnr = psnr_total / batch_count
    
    # Validation
    generator.eval()
    val_psnr = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for hr_images, _ in test_loader:
            hr_images = hr_images.to(device)
            lr_images = downsample_images(hr_images)
            _, _, h, w = hr_images.shape
            hr_images_upscaled = F.resize(hr_images, size=[h*2, w*2], 
                                        interpolation=transforms.InterpolationMode.BICUBIC)
            sr_images = generator(lr_images)
            val_psnr += PSNR(hr_images_upscaled, sr_images).item()
            val_batches += 1
    
    avg_val_psnr = val_psnr / val_batches
    
    # Update learning rates
    g_scheduler.step()
    d_scheduler.step()
    
    # Save best model with the _7 suffix
    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        torch.save(generator.state_dict(), "best_generator_sr_10.pth")
        print(f"New best PSNR: {best_psnr:.2f} dB - Model saved as best_generator_sr_10.pth")
    
    print(f"Epoch {epoch+1} Complete: Val PSNR: {avg_val_psnr:.2f} dB")
    
    # Record history
    history['d_loss'].append(d_loss_total / batch_count)
    history['g_loss'].append(g_loss_total / batch_count)
    history['psnr'].append(avg_psnr)
    history['val_psnr'].append(avg_val_psnr)
    
    # Print epoch summary
    print(f"Epoch {epoch + 1} Summary: "
          f"Train PSNR: {avg_psnr:.2f} dB, "
          f"Val PSNR: {avg_val_psnr:.2f} dB, "
          f"G Loss: {g_loss_total / batch_count:.4f}, "
          f"D Loss: {d_loss_total / batch_count:.4f}")
    
    # Early stopping if PSNR > 36
    if avg_val_psnr > 36:
        print(f"Achieved PSNR > 36 dB ({avg_val_psnr:.2f})! Stopping training.")
        break

# Final evaluation - loading best model
generator.load_state_dict(torch.load("best_generator_sr_10.pth"))
generator.eval()

test_psnr = 0.0
test_batches = 0

with torch.no_grad():
    for hr_images, _ in test_loader:
        hr_images = hr_images.to(device)
        lr_images = downsample_images(hr_images)
        _, _, h, w = hr_images.shape
        hr_images_upscaled = F.resize(hr_images, size=[h*2, w*2], 
                                interpolation=transforms.InterpolationMode.BICUBIC)
        sr_images = generator(lr_images)
        test_psnr += PSNR(hr_images_upscaled, sr_images).item()
        test_batches += 1
        
        if test_batches == 1:  # Save first batch for visualization
            test_lr = lr_images
            test_sr = sr_images
            test_hr = hr_images_upscaled

print(f"Final Test PSNR: {test_psnr/test_batches:.2f} dB")

# Visualization functions
def display_results(lr, sr, hr, num=5, save_path="sr_results_final_10.png"):
    fig, axes = plt.subplots(3, num, figsize=(20, 12))
    for i in range(num):
        # Low-res
        axes[0,i].imshow(lr[i].cpu().permute(1,2,0) * 0.5 + 0.5)
        axes[0,i].axis('off')
        if i == 0: axes[0,i].set_title('Low Resolution')
        
        # Super-resolved
        axes[1,i].imshow(sr[i].cpu().permute(1,2,0) * 0.5 + 0.5)
        axes[1,i].axis('off')
        if i == 0: axes[1,i].set_title('Super-Resolved')
        
        # High-res
        axes[2,i].imshow(hr[i].cpu().permute(1,2,0) * 0.5 + 0.5)
        axes[2,i].axis('off')
        if i == 0: axes[2,i].set_title('Original High-Res')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

display_results(test_lr, test_sr, test_hr, save_path="sr_results_final_10.png")

# Plot training curves - use _7 suffix for new plots
plt.figure(figsize=(12,6))
plt.plot(history['psnr'], label='Train PSNR')
plt.plot(history['val_psnr'], label='Validation PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.title('PSNR During Training (Extended)')
plt.savefig('training_psnr_curve_sr10.png')
plt.close()

plt.figure(figsize=(12,6))
plt.plot(history['g_loss'], label='Generator Loss')
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses (Extended)')
plt.savefig('training_loss_curve_sr10.png')
plt.close()