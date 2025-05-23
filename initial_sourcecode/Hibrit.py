import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense, 
    Add, Activation, UpSampling2D, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Tuz-biber gürültüsü ekleyen fonksiyon (RGB)
def add_salt_and_pepper_noise_rgb(images, prob=0.1):
    noisy_images = []
    for img in images:
        noisy_img = np.copy(img)
        # Tuz (salt) ekleme
        num_salt = np.ceil(prob * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        noisy_img[coords[0], coords[1], coords[2]] = 1

        # Biber (pepper) ekleme
        num_pepper = np.ceil(prob * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        noisy_img[coords[0], coords[1], coords[2]] = 0

        noisy_images.append(noisy_img)
    return np.array(noisy_images)

# PSNR Hesaplama
def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# Residual Block
def residual_block(x, filters, kernel_size=3):
    res = Conv2D(filters, kernel_size, padding='same')(x)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)
    res = Conv2D(filters, kernel_size, padding='same')(res)
    res = BatchNormalization()(res)
    res = Add()([x, res])
    res = Activation('relu')(res)
    return res

# ResNet Tabanlı Generator modeli
def build_resnet_generator(input_shape=(32, 32, 3), num_residual_blocks=5):
    inputs = Input(shape=input_shape)

    # İlk katman
    x = Conv2D(64, kernel_size=9, padding='same')(inputs)
    x = Activation('relu')(x)

    # Residual Blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, 64)

    # Son katman
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)

    # Inputs'u 64 kanala projekte et
    residual = Conv2D(64, kernel_size=1, padding='same')(inputs)

    # Residual Connection
    x = Add()([residual, x])
    x = Activation('relu')(x)

    # Çıkış katmanı
    outputs = Conv2D(3, kernel_size=9, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs, name="ResNet_Generator")

# Daha derin bir Discriminator modeli
def build_discriminator(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs, x, name="Discriminator")

# CIFAR-10 veri setini yükleme
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Verileri normalize etme (0-1 aralığına getirme)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Eğitim ve test verilerine gürültü ekleme
x_train_noisy = add_salt_and_pepper_noise_rgb(x_train, prob=0.05)
x_test_noisy = add_salt_and_pepper_noise_rgb(x_test, prob=0.05)

# Sonuçları kontrol etmek için örnek görüntü
def display_noisy_images(x_noisy, x_original, num=5):
    plt.figure(figsize=(15, 6))
    for i in range(num):
        # Gürültülü Görüntü
        plt.subplot(2, num, i + 1)
        plt.imshow(x_noisy[i])
        plt.axis('off')
        if i == 0:
            plt.title('Gürültülü')

        # Orijinal Görüntü
        plt.subplot(2, num, i + 1 + num)
        plt.imshow(x_original[i])
        plt.axis('off')
        if i == 0:
            plt.title('Orijinal')
    plt.tight_layout()
    plt.show()

# İlk 5 eğitim görüntüsünü göster
display_noisy_images(x_train_noisy, x_train)

# Modelleri oluşturma
generator = build_resnet_generator()
discriminator = build_discriminator()

# Optimizer'ları tanımla
generator_optimizer = Adam(0.0002, 0.5)
discriminator_optimizer = Adam(0.0002, 0.5)

# Kayıp fonksiyonları
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Eğitim fonksiyonu
@tf.function
def train_step(noisy_images, real_images):
    batch_size = tf.shape(real_images)[0]
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    # Generator'dan sahte görüntüler üret
    with tf.GradientTape(persistent=True) as tape:
        generated_images = generator(noisy_images, training=True)

        # Discriminator'ın gerçek ve sahte görüntüler için tahminleri
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Kayıp hesaplamaları
        d_loss_real = binary_cross_entropy(real_labels, real_output)
        d_loss_fake = binary_cross_entropy(fake_labels, fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = binary_cross_entropy(real_labels, fake_output)

    # Gradients'leri hesapla ve uygulama
    gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return d_loss, g_loss

# Eğitim döngüsü
epochs = 30  # Eğitim süresini ihtiyaca göre artırabilirsiniz
batch_size = 64

# Veri setini oluştur
buffer_size = x_train_noisy.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train)).shuffle(buffer_size).batch(batch_size)

history = {'d_loss': [], 'g_loss': []}

for epoch in range(epochs):
    d_loss_total = 0
    g_loss_total = 0
    batch_count = 0

    for noisy_batch, real_batch in train_dataset:
        d_loss, g_loss = train_step(noisy_batch, real_batch)
        d_loss_total += d_loss
        g_loss_total += g_loss
        batch_count += 1

    history['d_loss'].append(d_loss_total / batch_count)
    history['g_loss'].append(g_loss_total / batch_count)

    print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss_total / batch_count:.4f}, Generator Loss: {g_loss_total / batch_count:.4f}")

# Test verisiyle denoising
denoised_images = generator.predict(x_test_noisy)

# PSNR hesapla
psnr_values = PSNR(x_test, denoised_images)
print("Average PSNR:", np.mean(psnr_values.numpy()))

# Görüntüleme
def display_denoised_images(noisy, denoised, original, num=5):
    plt.figure(figsize=(15, 9))
    for i in range(num):
        # Gürültülü Görüntü
        plt.subplot(3, num, i + 1)
        plt.imshow(noisy[i])
        plt.axis('off')
        if i == 0:
            plt.title("Noisy")

        # Denoised Görüntü
        plt.subplot(3, num, i + 1 + num)
        plt.imshow(denoised[i])
        plt.axis('off')
        if i == 0:
            plt.title("Denoised")

        # Orijinal Görüntü
        plt.subplot(3, num, i + 1 + 2*num)
        plt.imshow(original[i])
        plt.axis('off')
        if i == 0:
            plt.title("Original")
    plt.tight_layout()
    plt.show()

# İlk 5 test görüntüsünü göster
display_denoised_images(x_test_noisy, denoised_images, x_test)

# Kayıp fonksiyonu grafiği
plt.figure(figsize=(12, 6))
plt.plot(history['d_loss'], label="Discriminator Loss")
plt.plot(history['g_loss'], label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")
plt.show()