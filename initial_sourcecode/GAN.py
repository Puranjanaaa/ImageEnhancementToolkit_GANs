kötü eğitilen 100 epokluk kod

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Tuz-biber gürültüsü ekleyen fonksiyon (grayscale)
def add_salt_and_pepper_noise(images, amount=0.05):
    noisy_images = images.copy()
    for i in range(images.shape[0]):
        num_salt = int(amount * images.shape[1] * images.shape[2])
        salt_coords = [np.random.randint(0, images.shape[1], num_salt), np.random.randint(0, images.shape[2], num_salt)]
        noisy_images[i][salt_coords[0], salt_coords[1], :] = 1

        num_pepper = int(amount * images.shape[1] * images.shape[2])
        pepper_coords = [np.random.randint(0, images.shape[1], num_pepper), np.random.randint(0, images.shape[2], num_pepper)]
        noisy_images[i][pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_images

# Tuz-biber gürültüsü ekleyen fonksiyon (RGB)
def add_salt_and_pepper_noise_rgb(images, prob=0.1):
    noisy_images = []
    for img in images:
        noisy_img = np.copy(img)
        num_pepper = int(prob * img.size * 0.5)
        num_salt = int(prob * img.size * 0.5)

        salt_coords = [np.random.randint(0, i, num_salt) for i in img.shape]
        noisy_img[salt_coords[0], salt_coords[1], salt_coords[2]] = 1

        pepper_coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
        noisy_img[pepper_coords[0], pepper_coords[1], pepper_coords[2]] = 0

        noisy_images.append(noisy_img)

    return np.array(noisy_images)

# PSNR Hesaplama
def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# CIFAR-10 veri setini yükleme
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Tuz-biber gürültüsü ekleme (RGB kullanımı)
x_train_noisy = add_salt_and_pepper_noise_rgb(x_train)
x_test_noisy = add_salt_and_pepper_noise_rgb(x_test)

# Görselleştirme
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.axis('off')
    plt.title("Original")

    plt.subplot(2, 5, i + 6)
    plt.imshow(x_train_noisy[i])
    plt.axis('off')
    plt.title("Noisy")

plt.tight_layout()
plt.show()

# Generator modeli
def build_generator():
    input_layer = Input(shape=(32, 32, 3))

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input_layer)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    output_layer = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    return Model(input_layer, output_layer)

# Discriminator modeli
def build_discriminator():
    input_layer = Input(shape=(32, 32, 3))

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(input_layer)
    x = LeakyReLU()(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    output_layer = Dense(1, activation='sigmoid')(x)

    return Model(input_layer, output_layer)

# Modelleri oluşturma
generator = build_generator()
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
epochs = 100
batch_size = 64
buffer_size = x_train_noisy.shape[0]

# Veri setini oluştur
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train)).shuffle(buffer_size).batch(batch_size)

history = {'loss': [], 'val_loss': []}

for epoch in range(epochs):
    d_loss_total = 0
    g_loss_total = 0
    batch_count = 0

    for noisy_batch, real_batch in train_dataset:
        d_loss, g_loss = train_step(noisy_batch, real_batch)
        d_loss_total += d_loss
        g_loss_total += g_loss
        batch_count += 1

    history['loss'].append(d_loss_total / batch_count)
    history['val_loss'].append(g_loss_total / batch_count)

    print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss_total / batch_count}, Generator Loss: {g_loss_total / batch_count}")

# Test verisiyle denoising
denoised_images = generator.predict(x_test_noisy)

# PSNR hesapla
psnr_values = PSNR(x_test, denoised_images)
print("Average PSNR:", np.mean(psnr_values))

# Görüntüleme
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test_noisy[i])
    plt.axis('off')
    plt.title("Noisy")

    plt.subplot(3, 5, i + 6)
    plt.imshow(denoised_images[i])
    plt.axis('off')
    plt.title("Denoised")

    plt.subplot(3, 5, i + 11)
    plt.imshow(x_test[i])
    plt.axis('off')
    plt.title("Original")

plt.tight_layout()
plt.show()

# Kayıp fonksiyonu grafiği
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label="Training Loss")
plt.plot(history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()