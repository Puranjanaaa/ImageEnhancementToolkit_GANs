import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs("sr_results", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced downsampling with realistic degradation
def downsample_images(images, scale_factor=2):
    """Create low-resolution images with realistic degradation pipeline"""
    # Add slight Gaussian blur
    blurred = F.gaussian_blur(images, kernel_size=[3, 3], sigma=[0.3, 1.2])
    
    # Downsample with bicubic interpolation
    _, _, h, w = images.shape
    low_res = F.resize(blurred, size=[h//scale_factor, w//scale_factor], 
                    interpolation=transforms.InterpolationMode.BICUBIC)
    
    # Add compression artifacts
    low_res = torch.clamp(low_res + torch.randn_like(low_res)*0.01, -1, 1)
    
    # Upsample back to original size
    return F.resize(low_res, size=[h, w], 
                  interpolation=transforms.InterpolationMode.BICUBIC)

# Model components
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        y = self.sigmoid(avg_out + max_out)
        return x * y

class RCAB(nn.Module):
    def __init__(self, channels, reduction=8):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            ChannelAttention(channels, reduction)
        )
    
    def forward(self, x):
        return x + self.body(x)

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

class EnhancedRCAB(nn.Module):
    def __init__(self, channels, reduction=8):
        super(EnhancedRCAB, self).__init__()
        self.channel_attention = RCAB(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualGroup(nn.Module):
    def __init__(self, n_blocks, channels):
        super(ResidualGroup, self).__init__()
        blocks = [EnhancedRCAB(channels) for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        self.blocks = nn.Sequential(*blocks)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.blocks(x))

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

class GeneratorSR(nn.Module):
    def __init__(self, scale_factor=2, n_groups=10, n_blocks=20, channels=128):
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
                DenseBlock(channels)
            ])
        self.upsample = nn.Sequential(*upsampling)
        
        # Output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, 3, 3, padding=1),
            nn.Tanh()
        )
    
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

# PSNR Calculation
def calculate_psnr(y_true, y_pred):
    y_true = torch.clamp(y_true, -1, 1)
    y_pred = torch.clamp(y_pred, -1, 1)
    
    # Convert from [-1,1] to [0,1]
    y_true = (y_true + 1) / 2
    y_pred = (y_pred + 1) / 2
    
    mse = torch.mean((y_true - y_pred) ** 2)
    return 10 * torch.log10(1.0 / mse)

# Load and process a CIFAR-10 image
def process_cifar_image(model, output_dir="sr_results"):
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    cifar_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Get a random image from the dataset
    idx = np.random.randint(0, len(cifar_dataset))
    img_tensor, _ = cifar_dataset[idx]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Create low-res version
    with torch.no_grad():
        # Create low-res image
        lr_img = downsample_images(img_tensor)
        
        # Generate super-resolved image
        sr_img = model(lr_img)
        
        # Create bicubic upscaled version for comparison
        _, _, h_sr, w_sr = sr_img.shape
        bicubic_img = F.resize(lr_img, size=[h_sr, w_sr], 
                              interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Calculate PSNR
        hr_upscaled = F.resize(img_tensor, size=[h_sr, w_sr], 
                             interpolation=transforms.InterpolationMode.BICUBIC)
        sr_psnr = calculate_psnr(hr_upscaled, sr_img).item()
        bicubic_psnr = calculate_psnr(hr_upscaled, bicubic_img).item()
        
        print(f"Super-resolution PSNR: {sr_psnr:.2f} dB")
        print(f"Bicubic upscaling PSNR: {bicubic_psnr:.2f} dB")
        print(f"PSNR Improvement: {sr_psnr - bicubic_psnr:.2f} dB")
        
        # Convert tensors back to images ([-1,1] to [0,1])
        original_img = (img_tensor + 1) / 2
        lr_img = (lr_img + 1) / 2
        bicubic_img = (bicubic_img + 1) / 2
        sr_img = (sr_img + 1) / 2
        
        # Save images
        img_name = f"cifar_{idx}"
        
        # Convert to PIL images and save
        to_pil = transforms.ToPILImage()
        
        # Original
        original_pil = to_pil(original_img.squeeze().cpu())
        original_pil.save(os.path.join(output_dir, f"{img_name}_original.png"))
        
        # Low-res
        lr_pil = to_pil(lr_img.squeeze().cpu())
        lr_pil.save(os.path.join(output_dir, f"{img_name}_lowres.png"))
        
        # Bicubic upscaled
        bicubic_pil = to_pil(bicubic_img.squeeze().cpu())
        bicubic_pil.save(os.path.join(output_dir, f"{img_name}_bicubic.png"))
        
        # Super-resolved
        sr_pil = to_pil(sr_img.squeeze().cpu())
        sr_pil.save(os.path.join(output_dir, f"{img_name}_superres.png"))
        
        # Create comparison visualization
        plt.figure(figsize=(16, 12))
        
        # Original
        plt.subplot(2, 2, 1)
        plt.imshow(original_img.squeeze().cpu().permute(1, 2, 0))
        plt.title('Original Image')
        plt.axis('off')
        
        # Low-res
        plt.subplot(2, 2, 2)
        plt.imshow(lr_img.squeeze().cpu().permute(1, 2, 0))
        plt.title('Low Resolution')
        plt.axis('off')
        
        # Bicubic upscaled
        plt.subplot(2, 2, 3)
        plt.imshow(bicubic_img.squeeze().cpu().permute(1, 2, 0))
        plt.title(f'Bicubic Upscaled (PSNR: {bicubic_psnr:.2f} dB)')
        plt.axis('off')
        
        # Super-resolved
        plt.subplot(2, 2, 4)
        plt.imshow(sr_img.squeeze().cpu().permute(1, 2, 0))
        plt.title(f'Super-Resolved (PSNR: {sr_psnr:.2f} dB)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{img_name}_comparison.png"))
        plt.close()
        
        print(f"Results saved to {output_dir} directory.")
        
        return original_img, lr_img, bicubic_img, sr_img

# Main execution
if __name__ == "__main__":
    # Initialize and load the trained model
    generator = GeneratorSR(scale_factor=2, n_groups=8, n_blocks=16, channels=128).to(device)
    
    # Try to load the model, with error handling
    model_path = "models/best_generator_sr_9.pth"
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and has the correct path.")
        exit(1)
    
    generator.eval()
    
    # Process a random CIFAR-10 image
    print("Processing a random CIFAR-10 image...")
    process_cifar_image(generator)
    
    print("Super-resolution complete!")