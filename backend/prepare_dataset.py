import pandas as pd
import shutil
import os
from pathlib import Path
import random

# Veri seti yolları
SOURCE_IMAGES_DIR = "fashion_product_images_small/images"
STYLES_CSV = "fashion_product_images_small/styles.csv"
TARGET_DIR = "fashion_dataset_organized"

def create_directory_structure():
    """Hedef klasör yapısını oluştur"""
    # Ana kategoriler için klasörler
    master_categories = ['Apparel', 'Accessories']
    sub_categories = ['Topwear', 'Bottomwear', 'Watches', 'Footwear', 'Bags', 'Sunglasses']
    article_types = ['Shirts', 'Tshirts', 'Jeans', 'Trousers', 'Watches', 'Shoes', 'Handbags', 'Sunglasses']
    
    # Hedef klasörü oluştur
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)
    
    # Her kategori için klasörler oluştur
    for category in master_categories:
        os.makedirs(os.path.join(TARGET_DIR, 'masterCategory', category), exist_ok=True)
    
    for category in sub_categories:
        os.makedirs(os.path.join(TARGET_DIR, 'subCategory', category), exist_ok=True)
    
    for category in article_types:
        os.makedirs(os.path.join(TARGET_DIR, 'articleType', category), exist_ok=True)

def copy_images(df, category_type, category_name, image_ids, target_dir):
    """Seçilen görüntüleri hedef klasöre kopyala"""
    target_path = os.path.join(TARGET_DIR, category_type, category_name)
    os.makedirs(target_path, exist_ok=True)
    
    for img_id in image_ids:
        source_path = os.path.join(SOURCE_IMAGES_DIR, f"{img_id}.jpg")
        if os.path.exists(source_path):
            shutil.copy2(source_path, os.path.join(target_path, f"{img_id}.jpg"))

def prepare_dataset():
    """Veri setini hazırla"""
    try:
        # CSV dosyasını oku - hata toleranslı modda
        df = pd.read_csv(STYLES_CSV, 
                        encoding='utf-8',
                        on_bad_lines='skip',  # Hatalı satırları atla
                        quoting=1,  # QUOTE_ALL modu
                        escapechar='\\')  # Kaçış karakteri
        
        print(f"CSV dosyası başarıyla okundu. Toplam {len(df)} satır.")
        
        # Klasör yapısını oluştur
        create_directory_structure()
        
        # Her kategori için 200 örnek seç
        samples_per_category = 200
        
        # MasterCategory için örnekler
        for category in df['masterCategory'].unique():
            category_samples = df[df['masterCategory'] == category].sample(
                n=min(samples_per_category, len(df[df['masterCategory'] == category])),
                random_state=42
            )
            copy_images(df, 'masterCategory', category, category_samples['id'].tolist(), TARGET_DIR)
            print(f"MasterCategory - {category}: {len(category_samples)} örnek kopyalandı")
        
        # SubCategory için örnekler
        for category in df['subCategory'].unique():
            category_samples = df[df['subCategory'] == category].sample(
                n=min(samples_per_category, len(df[df['subCategory'] == category])),
                random_state=42
            )
            copy_images(df, 'subCategory', category, category_samples['id'].tolist(), TARGET_DIR)
            print(f"SubCategory - {category}: {len(category_samples)} örnek kopyalandı")
        
        # ArticleType için örnekler
        for category in df['articleType'].unique():
            category_samples = df[df['articleType'] == category].sample(
                n=min(samples_per_category, len(df[df['articleType'] == category])),
                random_state=42
            )
            copy_images(df, 'articleType', category, category_samples['id'].tolist(), TARGET_DIR)
            print(f"ArticleType - {category}: {len(category_samples)} örnek kopyalandı")
            
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    prepare_dataset()
    print("Veri seti hazırlama tamamlandı!") 