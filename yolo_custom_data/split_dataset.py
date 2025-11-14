import os
import random
import shutil

# --- Configuración de las rutas ---
# Ruta a la carpeta que contiene las carpetas 'images' y 'labels' descargadas
source_dir = '/home/david/yolo_custom_data/export'
# Ruta donde se crearán las carpetas 'train', 'valid' y 'test'
output_dir = '/home/david/yolo_custom_data/dataset'

# --- Porcentajes de división ---
split_ratios = {
    'train': 0.7,
    'valid': 0.2,
    'test': 0.1 
}


# --- Rutas de origen ---
source_images_dir = os.path.join(source_dir, 'images')
source_labels_dir = os.path.join(source_dir, 'labels')

# 1. Crear las carpetas de destino si no existen
print(" Creando estructura de carpetas 'train', 'valid', 'test'...")
for split_name in split_ratios.keys():
    os.makedirs(os.path.join(output_dir, split_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split_name, 'labels'), exist_ok=True)

# 2. Obtener la lista de todos los archivos de imágenes y barajarlos
print(" shuffling images for random distribution...")
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)
print(f"Total de imágenes encontradas: {len(image_files)}")

# 3. Calcular los puntos de división
total_files = len(image_files)
train_split_idx = int(total_files * split_ratios['train'])
valid_split_idx = int(total_files * (split_ratios['train'] + split_ratios['valid']))

# 4. Asignar archivos a cada conjunto
train_files = image_files[:train_split_idx]
valid_files = image_files[train_split_idx:valid_split_idx]
test_files = image_files[valid_split_idx:]

file_splits = {
     'train': train_files,
     'valid': valid_files,
     'test': test_files
}

# 5. Mover los archivos a sus carpetas correspondientes
for split_name, files in file_splits.items():
    print(f"\n Moviendo {len(files)} archivos a la carpeta '{split_name}'...")
    for image_filename in files:
        base_filename = os.path.splitext(image_filename)[0]
        label_filename = f"{base_filename}.txt"

        source_image_path = os.path.join(source_images_dir, image_filename)
        source_label_path = os.path.join(source_labels_dir, label_filename)

        dest_image_path = os.path.join(output_dir, split_name, 'images', image_filename)
        dest_label_path = os.path.join(output_dir, split_name, 'labels', label_filename)

        shutil.move(source_image_path, dest_image_path)
                                                                                            
        if os.path.exists(source_label_path):
            shutil.move(source_label_path, dest_label_path)

print("\n ¡Distribución completada exitosamente!")
print(f"   - Archivos de entrenamiento: {len(train_files)}")
print(f"   - Archivos de validación: {len(valid_files)}")
print(f"   - Archivos de prueba: {len(test_files)}")
