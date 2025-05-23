import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Salt and pepper noise function (RGB)
def add_salt_and_pepper_noise_rgb(images, prob=0.3):
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

# ResNet-based Generator
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        # Initial layer
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(64))
        
        # Mid layer
        self.mid_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Additional layer
        self.additional_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # New layer
        self.new_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Project input to 64 channels
        self.residual_conv = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Sigmoid()
        )
        
        # Define ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        initial_out = self.initial(x)
        res_out = self.res_blocks(initial_out)
        mid_out = self.mid_layer(res_out)
        additional_out = self.additional_layer(mid_out)  # Pass through the additional layer
        new_out = self.new_layer(additional_out)  # Pass through the new layer
        
        # Add residual connection
        residual = self.residual_conv(x)
        out = new_out + residual
        out = self.relu(out)
        
        return self.output(out)

# Deeper Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            # Additional Block
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            # Output
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, 1),
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

# Display noisy images
def display_noisy_images(noisy, original, num=5):
    fig, axes = plt.subplots(2, num, figsize=(15, 6))
    for i in range(num):
        # Noisy image
        axes[0, i].imshow(noisy[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Noisy')
        
        # Original image
        axes[1, i].imshow(original[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Original')
    plt.tight_layout()
    plt.savefig(f"noisy_{num}.png") 
    # plt.show()

# Get a batch of training data
data_iter = iter(train_loader)
images, _ = next(data_iter)

# Add noise and display
noisy_images = add_salt_and_pepper_noise_rgb(images)
display_noisy_images(noisy_images, images)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Add this before the training loop
best_psnr = 0.0  # Initialize the best PSNR value
best_model_path = "best_generator_denoised_3.pth"  # Path to save the best model

# Training loop
epochs = 100
history = {'d_loss': [], 'g_loss': []}

print("Starting training...")
for epoch in range(epochs):
    d_loss_total = 0.0
    g_loss_total = 0.0
    batch_count = 0
    
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Add noise to create noisy images
        noisy_images = add_salt_and_pepper_noise_rgb(real_images, prob=0.05)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        discriminator_optimizer.zero_grad()
        
        # Real images
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake images
        generated_images = generator(noisy_images)
        fake_output = discriminator(generated_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        discriminator_optimizer.step()
        
        # -----------------
        # Train Generator
        # -----------------
        generator_optimizer.zero_grad()
        
        # Generate images and get discriminator output
        generated_images = generator(noisy_images)
        fake_output = discriminator(generated_images)
        
        # Generator loss
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        generator_optimizer.step()
        
        # Record losses
        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()
        batch_count += 1
    
    # Calculate average losses
    avg_d_loss = d_loss_total / batch_count
    avg_g_loss = g_loss_total / batch_count
    history['d_loss'].append(avg_d_loss)
    history['g_loss'].append(avg_g_loss)
    
    # Evaluate PSNR on the test set
    generator.eval()
    psnr_values = []
    with torch.no_grad():
        for real_images, _ in test_loader:
            real_images = real_images.to(device)
            noisy_images = add_salt_and_pepper_noise_rgb(real_images, prob=0.05)
            denoised_images = generator(noisy_images)
            psnr = PSNR(real_images, denoised_images)
            psnr_values.append(psnr.item())
    avg_psnr = np.mean(psnr_values)
    print(f"Epoch [{epoch+1}/{epochs}], d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
    
    # Save the best model
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(generator.state_dict(), best_model_path)
        print(f"New best model saved with PSNR: {best_psnr:.2f} dB")
    generator.train()

# Test the model
generator.eval()
psnr_values = []

with torch.no_grad():
    for i, (real_images, _) in enumerate(test_loader):
        real_images = real_images.to(device)
        noisy_images = add_salt_and_pepper_noise_rgb(real_images, prob=0.4)
        denoised_images = generator(noisy_images)
        
        # Calculate PSNR
        psnr = PSNR(real_images, denoised_images)
        psnr_values.append(psnr.item())
        
        if i == 0:  # Save first batch for visualization
            test_noisy = noisy_images
            test_denoised = denoised_images
            test_real = real_images
            break

print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")

# Display results
# Save results
def display_denoised_images(noisy, denoised, original, num=5, save_path="denoised_results_3.png"):
    fig, axes = plt.subplots(3, num, figsize=(15, 9))
    for i in range(num):
        # Noisy image
        axes[0, i].imshow(noisy[i].cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Noisy')
        
        # Denoised image
        axes[1, i].imshow(denoised[i].cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Denoised')
        
        # Original image
        axes[2, i].imshow(original[i].cpu().permute(1, 2, 0))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Original')
    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure
    print(f"Results saved to {save_path}")

display_denoised_images(test_noisy, test_denoised, test_real)

# Save training history plot
plt.figure(figsize=(12, 6))
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.savefig("training_loss_denoised_3.png")  # Save the figure
print("Training loss plot saved to 'training_loss_denoised.png'")