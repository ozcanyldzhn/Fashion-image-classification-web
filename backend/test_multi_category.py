import torch
from torchvision import transforms
from PIL import Image
import os
from train_multi_category import MultiCategoryModel

def load_model(category_type):
    """Belirtilen kategori tipi için modeli yükle"""
    # Kategori klasörlerini al
    categories = sorted(os.listdir(os.path.join('fashion_dataset_organized', category_type)))
    
    # Modeli oluştur ve ağırlıkları yükle
    model = MultiCategoryModel(len(categories))
    model.load_state_dict(torch.load(f'model/{category_type}_model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    return model, categories

def predict_image(image_path, category_type):
    """Görüntüyü sınıflandır"""
    # Modeli ve kategorileri yükle
    model, categories = load_model(category_type)
    
    # Görüntüyü yükle ve dönüştür
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Sonuçları döndür
    predicted_category = categories[predicted.item()]
    confidence = probabilities[predicted].item() * 100
    
    return {
        'category': predicted_category,
        'confidence': confidence,
        'all_probabilities': {cat: prob.item() * 100 for cat, prob in zip(categories, probabilities)}
    }

def test_all_categories(image_path):
    """Görüntüyü tüm kategori tipleri için sınıflandır"""
    results = {}
    for category_type in ['masterCategory', 'subCategory', 'articleType']:
        results[category_type] = predict_image(image_path, category_type)
    
    return results

if __name__ == "__main__":
    # Test görüntüsü yolu
    test_image = "fashion_dataset_organized/masterCategory/Apparel/1.jpg"  # Örnek bir görüntü
    
    # Tüm kategoriler için tahmin yap
    results = test_all_categories(test_image)
    
    # Sonuçları yazdır
    print("\nTahmin Sonuçları:")
    print("-" * 50)
    for category_type, result in results.items():
        print(f"\n{category_type}:")
        print(f"Tahmin: {result['category']}")
        print(f"Güven: {result['confidence']:.2f}%")
        print("\nTüm Olasılıklar:")
        for cat, prob in result['all_probabilities'].items():
            print(f"{cat}: {prob:.2f}%") 