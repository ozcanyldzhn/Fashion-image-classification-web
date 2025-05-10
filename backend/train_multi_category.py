import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FashionDataset(Dataset):
    def __init__(self, root_dir, category_type, transform=None):
        self.root_dir = root_dir
        self.category_type = category_type
        self.transform = transform
        self.categories = sorted(os.listdir(os.path.join(root_dir, category_type)))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        self.images = []
        self.labels = []
        
        # Her kategoriden görüntüleri topla
        for category in self.categories:
            category_path = os.path.join(root_dir, category_type, category)
            for img_name in os.listdir(category_path):
                if img_name.endswith('.jpg'):
                    self.images.append(os.path.join(category_path, img_name))
                    self.labels.append(self.category_to_idx[category])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MultiCategoryModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiCategoryModel, self).__init__()
        # ResNet50 modelini kullan (daha derin ve güçlü)
        self.model = models.resnet50(pretrained=True)
        
        # Son tam bağlantılı katmanı değiştir
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(category_type, num_epochs=30, batch_size=16, learning_rate=0.0001):
    # Veri dönüşümleri - Eğitim için veri artırma
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Doğrulama için basit dönüşümler
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Veri setlerini yükle
    train_dataset = FashionDataset('fashion_dataset_organized', category_type, transform=train_transform)
    val_dataset = FashionDataset('fashion_dataset_organized', category_type, transform=val_transform)
    
    # Veri setini eğitim ve doğrulama olarak böl
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    # Modeli oluştur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiCategoryModel(len(train_dataset.dataset.categories)).to(device)
    
    # Kayıp fonksiyonu ve optimizasyon
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    # Eğitim döngüsü
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Accuracy: {train_acc:.2f}%')
        
        # Doğrulama
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%')
        
        # Öğrenme oranını güncelle
        scheduler.step(val_acc)
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'model/{category_type}_model.pt')
            print(f'Model kaydedildi! (Validation Accuracy: {val_acc:.2f}%)')

if __name__ == "__main__":
    # Her kategori tipi için modeli eğit
    category_types = ['masterCategory', 'subCategory', 'articleType']
    for category_type in category_types:
        print(f"\nEğitim başlıyor: {category_type}")
        train_model(category_type)
        print(f"{category_type} eğitimi tamamlandı!") 