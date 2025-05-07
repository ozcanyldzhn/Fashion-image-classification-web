import os
import struct
from PIL import Image

# FashionMNIST dosyalarının bulunduğu klasör
DATA_DIR = "fashionmsit"

def read_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = f.read()
        images = list(images)
        images = [images[i * rows * cols:(i + 1) * rows * cols] for i in range(num)]
    return images

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = f.read()
        labels = list(labels)
    return labels

def save_images(images, labels, output_dir):
    for i, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        img = Image.new('L', (28, 28))  # grayscale (L)
        img.putdata(image)
        img_path = os.path.join(label_dir, f"{i}.png")
        img.save(img_path)

# Dosya yolları
train_image_file = os.path.join(DATA_DIR, 'train-images-idx3-ubyte')
train_label_file = os.path.join(DATA_DIR, 'train-labels-idx1-ubyte')
test_image_file = os.path.join(DATA_DIR, 't10k-images-idx3-ubyte')
test_label_file = os.path.join(DATA_DIR, 't10k-labels-idx1-ubyte')

# Veriyi oku
train_images = read_images(train_image_file)
train_labels = read_labels(train_label_file)
test_images = read_images(test_image_file)
test_labels = read_labels(test_label_file)

# Çıktı klasörleri
train_output_dir = os.path.join(DATA_DIR, 'output', 'train')
test_output_dir = os.path.join(DATA_DIR, 'output', 'test')

# Klasörleri oluştur ve görselleri kaydet
save_images(train_images, train_labels, train_output_dir)
save_images(test_images, test_labels, test_output_dir)

print("Veriler başarıyla PNG formatına dönüştürüldü ve klasörlere ayrıldı.")
