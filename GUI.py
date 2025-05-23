import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QTabWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import random
from PIL import Image
import os

# ==================== Super Resolution Model ====================
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

# ==================== Deblurring Model ====================
class ResidualBlockDeblur(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockDeblur, self).__init__()
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

class GeneratorDeblur(nn.Module):
    def __init__(self, num_residual_blocks=12):
        super(GeneratorDeblur, self).__init__()
        
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
            self.res_blocks.append(ResidualBlockDeblur(512))
        
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

# ==================== Denoising Model ====================
class ResidualBlockDenoise(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockDenoise, self).__init__()
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

class GeneratorDenoise(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(GeneratorDenoise, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlockDenoise(64))
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

# ==================== Utility Functions ====================
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

def add_random_blur(images, max_kernel_size=5, max_sigma=2.0):
    blurred_images = torch.zeros_like(images)
    for i in range(images.shape[0]):
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.7, max_sigma)
        blurred_images[i] = F.gaussian_blur(images[i], kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blurred_images

def add_salt_and_pepper_noise_rgb(images, prob=0.2):
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

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def calculate_psnr(y_true, y_pred):
    y_true = torch.clamp(y_true, -1, 1)
    y_pred = torch.clamp(y_pred, -1, 1)
    
    # Convert from [-1,1] to [0,1]
    y_true = (y_true + 1) / 2
    y_pred = (y_pred + 1) / 2
    
    mse = torch.mean((y_true - y_pred) ** 2)
    return 10 * torch.log10(1.0 / mse)

# ==================== Main Application ====================
class ImageEnhancementToolkit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Enhancement Toolkit (Generative Models)")
        self.setGeometry(100, 100, 900, 600)
        
        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Super Resolution model
        self.generator_sr = GeneratorSR(scale_factor=2, n_groups=8, n_blocks=16, channels=128).to(self.device)
        try:
            self.generator_sr.load_state_dict(torch.load("models/best_generator_sr_9.pth", map_location=self.device))
            print("Super-resolution model loaded successfully")
        except Exception as e:
            print(f"Error loading super-resolution model: {e}")
            exit(1)
        self.generator_sr.eval()
        
        # Deblurring model
        self.generator_deblur = GeneratorDeblur().to(self.device)
        try:
            self.generator_deblur.load_state_dict(torch.load("models/best_generator_deblur.pth", map_location=self.device))
            print("Deblurring model loaded successfully")
        except Exception as e:
            print(f"Error loading deblurring model: {e}")
        self.generator_deblur.eval()
        
        # Denoising model
        self.generator_denoise = GeneratorDenoise().to(self.device)
        try:
            self.generator_denoise.load_state_dict(torch.load("models/best_generator_denoised_3.pth", map_location=self.device))
            print("Denoising model loaded successfully")
        except Exception as e:
            print(f"Error loading denoising model: {e}")
        self.generator_denoise.eval()
        
        # Load CIFAR-10 test set
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For SR
        ])
        self.transform_deblur = transforms.Compose([
            transforms.ToTensor(),  # For deblurring
        ])
        self.transform_denoise = transforms.Compose([
            transforms.ToTensor(),  # For denoising
        ])
        
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)
        
        self.test_dataset_deblur = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_deblur)
        self.test_loader_deblur = DataLoader(self.test_dataset_deblur, batch_size=1, shuffle=True)
        
        self.test_dataset_denoise = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_denoise)
        self.test_loader_denoise = DataLoader(self.test_dataset_denoise, batch_size=1, shuffle=True)
        
        # Initialize variables
        self.current_image = None
        self.low_res_image = None
        self.sr_image = None
        self.blurred_image = None
        self.deblurred_image = None
        self.noisy_image = None
        self.denoised_image = None
        
        # Create UI
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Super Resolution Tab
        self.sr_tab = QWidget()
        self.tabs.addTab(self.sr_tab, "Super Resolution")
        self.initSRTab()
        
        # Deblurring Tab
        self.deblur_tab = QWidget()
        self.tabs.addTab(self.deblur_tab, "Deblurring")
        self.initDeblurTab()
        
        # Denoising Tab
        self.denoise_tab = QWidget()
        self.tabs.addTab(self.denoise_tab, "Denoising")
        self.initDenoiseTab()
        
    def initSRTab(self):
        layout = QVBoxLayout()
        self.sr_tab.setLayout(layout)
        
        # Image display area
        self.sr_image_layout = QHBoxLayout()
        layout.addLayout(self.sr_image_layout)
        
        # Original image
        self.sr_original_label = QLabel("Original Image")
        self.sr_original_label.setAlignment(Qt.AlignCenter)
        self.sr_original_label.setMinimumSize(256, 256)
        self.sr_image_layout.addWidget(self.sr_original_label)
        
        # Low-res image
        self.low_res_label = QLabel("Degraded Image")
        self.low_res_label.setAlignment(Qt.AlignCenter)
        self.low_res_label.setMinimumSize(256, 256)
        self.sr_image_layout.addWidget(self.low_res_label)
        
        # Super-resolved image
        self.sr_result_label = QLabel("Super-Resolved Image")
        self.sr_result_label.setAlignment(Qt.AlignCenter)
        self.sr_result_label.setMinimumSize(256, 256)
        self.sr_image_layout.addWidget(self.sr_result_label)
        
        # PSNR label
        self.sr_psnr_label = QLabel("PSNR: -- dB")
        self.sr_psnr_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sr_psnr_label)
        
        # Button area
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        # Load CIFAR image button
        self.sr_cifar_button = QPushButton("Load Random CIFAR-10 Image")
        self.sr_cifar_button.clicked.connect(self.load_sr_cifar_image)
        button_layout.addWidget(self.sr_cifar_button)
        
        # Process button
        self.sr_process_button = QPushButton("Super Resolve")
        self.sr_process_button.clicked.connect(self.process_sr_image)
        self.sr_process_button.setEnabled(False)
        button_layout.addWidget(self.sr_process_button)

    def initDeblurTab(self):
        layout = QVBoxLayout()
        self.deblur_tab.setLayout(layout)
        
        # Image display area
        self.deblur_image_layout = QHBoxLayout()
        layout.addLayout(self.deblur_image_layout)
        
        # Original image
        self.deblur_original_label = QLabel("Original Image")
        self.deblur_original_label.setAlignment(Qt.AlignCenter)
        self.deblur_original_label.setMinimumSize(256, 256)
        self.deblur_image_layout.addWidget(self.deblur_original_label)
        
        # Blurred image
        self.blurred_label = QLabel("Blurred Image")
        self.blurred_label.setAlignment(Qt.AlignCenter)
        self.blurred_label.setMinimumSize(256, 256)
        self.deblur_image_layout.addWidget(self.blurred_label)
        
        # Deblurred image
        self.deblur_result_label = QLabel("Deblurred Image")
        self.deblur_result_label.setAlignment(Qt.AlignCenter)
        self.deblur_result_label.setMinimumSize(256, 256)
        self.deblur_image_layout.addWidget(self.deblur_result_label)
        
        # PSNR label
        self.deblur_psnr_label = QLabel("PSNR: -- dB")
        self.deblur_psnr_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.deblur_psnr_label)
        
        # Button area
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        # Load CIFAR image button
        self.deblur_cifar_button = QPushButton("Load Random CIFAR-10 Image")
        self.deblur_cifar_button.clicked.connect(self.load_deblur_cifar_image)
        button_layout.addWidget(self.deblur_cifar_button)
        
        # Process button
        self.deblur_process_button = QPushButton("Deblur")
        self.deblur_process_button.clicked.connect(self.process_deblur_image)
        self.deblur_process_button.setEnabled(False)
        button_layout.addWidget(self.deblur_process_button)

    def initDenoiseTab(self):
        layout = QVBoxLayout()
        self.denoise_tab.setLayout(layout)
        
        # Image display area
        self.denoise_image_layout = QHBoxLayout()
        layout.addLayout(self.denoise_image_layout)
        
        # Original image
        self.denoise_original_label = QLabel("Original Image")
        self.denoise_original_label.setAlignment(Qt.AlignCenter)
        self.denoise_original_label.setMinimumSize(256, 256)
        self.denoise_image_layout.addWidget(self.denoise_original_label)
        
        # Noisy image
        self.noisy_label = QLabel("Noisy Image")
        self.noisy_label.setAlignment(Qt.AlignCenter)
        self.noisy_label.setMinimumSize(256, 256)
        self.denoise_image_layout.addWidget(self.noisy_label)
        
        # Denoised image
        self.denoise_result_label = QLabel("Denoised Image")
        self.denoise_result_label.setAlignment(Qt.AlignCenter)
        self.denoise_result_label.setMinimumSize(256, 256)
        self.denoise_image_layout.addWidget(self.denoise_result_label)
        
        # PSNR label
        self.psnr_label = QLabel("PSNR: -- dB")
        self.psnr_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.psnr_label)
        
        # Button area
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        # Load CIFAR image button
        self.denoise_cifar_button = QPushButton("Load Random CIFAR-10 Image")
        self.denoise_cifar_button.clicked.connect(self.load_denoise_cifar_image)
        button_layout.addWidget(self.denoise_cifar_button)
        
        # Process button
        self.denoise_process_button = QPushButton("Denoise")
        self.denoise_process_button.clicked.connect(self.process_denoise_image)
        self.denoise_process_button.setEnabled(False)
        button_layout.addWidget(self.denoise_process_button)

    # ==================== Super Resolution Functions ====================
    def load_sr_cifar_image(self):
        # Get a random image from CIFAR-10
        test_image, _ = next(iter(self.test_loader))
        self.current_image = test_image.to(self.device)
        
        # Create low-res version with realistic degradation
        self.low_res_image = downsample_images(self.current_image)
        
        # Display images
        self.display_sr_images()
        self.sr_process_button.setEnabled(True)
        
    def process_sr_image(self):
        try:
            if self.low_res_image is not None:
                with torch.no_grad():
                    # Generate the super-resolved image
                    sr_image = self.generator_sr(self.low_res_image)
                    self.sr_image = sr_image

                    # Ensure the original image matches the SR image dimensions
                    _, _, h, w = self.sr_image.shape
                    resized_original = F.resize(self.current_image, size=[h, w], 
                                            interpolation=transforms.InterpolationMode.BICUBIC)

                    # Calculate PSNR between resized original and SR image
                    psnr = calculate_psnr(resized_original, self.sr_image)
                    self.sr_psnr_label.setText(f"PSNR: {psnr.item():.2f} dB")

                    # Display the images
                    self.display_sr_images()
            else:
                print("Low-resolution image is not loaded.")
        except Exception as e:
            print(f"Error in process_sr_image: {e}")
            
    def display_sr_images(self):
        # Display original image
        if self.current_image is not None:
            original_np = denormalize(self.current_image[0].cpu()).numpy().transpose(1, 2, 0)
            original_qimage = self.numpy_to_qimage(original_np)
            self.sr_original_label.setPixmap(QPixmap.fromImage(original_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display low-res image
        if self.low_res_image is not None:
            low_res_np = denormalize(self.low_res_image[0].cpu()).numpy().transpose(1, 2, 0)
            low_res_qimage = self.numpy_to_qimage(low_res_np)
            self.low_res_label.setPixmap(QPixmap.fromImage(low_res_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display super-resolved image
        if self.sr_image is not None:
            sr_np = denormalize(self.sr_image[0].cpu()).numpy().transpose(1, 2, 0)
            sr_qimage = self.numpy_to_qimage(sr_np)
            self.sr_result_label.setPixmap(QPixmap.fromImage(sr_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    # ==================== Deblurring Functions ====================
    def load_deblur_cifar_image(self):
        # Get a random image from CIFAR-10
        test_image, _ = next(iter(self.test_loader_deblur))
        self.current_image_deblur = test_image.to(self.device)
        
        # Create blurred version
        self.blurred_image = add_random_blur(self.current_image_deblur)
        
        # Display images
        self.display_deblur_images()
        self.deblur_process_button.setEnabled(True)
        
    def process_deblur_image(self):
        try:
            if self.blurred_image is not None:
                with torch.no_grad():
                    deblurred_image = self.generator_deblur(self.blurred_image)
                    self.deblurred_image = deblurred_image

                    # Calculate PSNR
                    psnr = calculate_psnr(self.current_image_deblur, self.deblurred_image)
                    self.deblur_psnr_label.setText(f"PSNR: {psnr.item():.2f} dB")

                    self.display_deblur_images()
        except Exception as e:
            print(f"Error in process_deblur_image: {e}")
        
    def display_deblur_images(self):
        # Display original image
        if self.current_image_deblur is not None:
            original_np = self.current_image_deblur[0].cpu().numpy().transpose(1, 2, 0)
            original_qimage = self.numpy_to_qimage(original_np)
            self.deblur_original_label.setPixmap(QPixmap.fromImage(original_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display blurred image
        if self.blurred_image is not None:
            blurred_np = self.blurred_image[0].cpu().numpy().transpose(1, 2, 0)
            blurred_qimage = self.numpy_to_qimage(blurred_np)
            self.blurred_label.setPixmap(QPixmap.fromImage(blurred_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display deblurred image
        if self.deblurred_image is not None:
            deblurred_np = self.deblurred_image[0].cpu().numpy().transpose(1, 2, 0)
            deblurred_qimage = self.numpy_to_qimage(deblurred_np)
            self.deblur_result_label.setPixmap(QPixmap.fromImage(deblurred_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    # ==================== Denoising Functions ====================
    def load_denoise_cifar_image(self):
        # Get a random image from CIFAR-10
        test_image, _ = next(iter(self.test_loader_denoise))
        self.current_image_denoise = test_image.to(self.device)
        
        # Create noisy version
        self.noisy_image = add_salt_and_pepper_noise_rgb(self.current_image_denoise, prob=0.2)
        
        # Display images
        self.display_denoise_images()
        self.denoise_process_button.setEnabled(True)
        
    def process_denoise_image(self):
        if self.noisy_image is not None:
            with torch.no_grad():
                denoised_image = self.generator_denoise(self.noisy_image)
            
            self.denoised_image = denoised_image
            
            # Calculate PSNR
            psnr = calculate_psnr(self.current_image_denoise, self.denoised_image)
            self.psnr_label.setText(f"PSNR: {psnr.item():.2f} dB")
            
            self.display_denoise_images()
        
    def display_denoise_images(self):
        # Display original image
        if self.current_image_denoise is not None:
            original_np = self.current_image_denoise[0].cpu().numpy().transpose(1, 2, 0)
            original_qimage = self.numpy_to_qimage(original_np)
            self.denoise_original_label.setPixmap(QPixmap.fromImage(original_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display noisy image
        if self.noisy_image is not None:
            noisy_np = self.noisy_image[0].cpu().numpy().transpose(1, 2, 0)
            noisy_qimage = self.numpy_to_qimage(noisy_np)
            self.noisy_label.setPixmap(QPixmap.fromImage(noisy_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display denoised image
        if self.denoised_image is not None:
            denoised_np = self.denoised_image[0].cpu().numpy().transpose(1, 2, 0)
            denoised_qimage = self.numpy_to_qimage(denoised_np)
            self.denoise_result_label.setPixmap(QPixmap.fromImage(denoised_qimage).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    # ==================== Utility Functions ====================
    def numpy_to_qimage(self, np_image):
        # Convert numpy array to QImage
        np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
        height, width, channel = np_image.shape
        # Convert numpy array to bytes
        bytes_img = np_image.tobytes()
        return QImage(bytes_img, width, height, width * 3, QImage.Format_RGB888)

# ==================== Main Execution ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEnhancementToolkit()
    window.show()
    sys.exit(app.exec_())