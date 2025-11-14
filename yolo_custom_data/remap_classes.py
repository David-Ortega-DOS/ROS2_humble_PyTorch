import os

# --- CONFIGURACIÓN ---

# 1. Define la ruta a la carpeta principal del dataset
# Esta carpeta debe contener 'train', 'valid' y 'test'
dataset_dir = '/home/david/yolo_custom_data/dataset'

# 2. Define las clases originales y las nuevas clases
original_names = [
            'biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 
                'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 
                    'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck'
                    ]

new_names = ['car', 'biker', 'pedestrian', 'trafficsignal']

# 3. Crea el mapa de conversión de ID de clase
# El valor será el NUEVO índice de la clase
remapping = {
    original_names.index('biker'): new_names.index('biker'),                 # 0 -> 1
    original_names.index('car'): new_names.index('car'),                     # 1 -> 0
    original_names.index('pedestrian'): new_names.index('pedestrian'),         # 2 -> 2
    original_names.index('trafficLight'): new_names.index('trafficsignal'),   # 3 -> 3
    original_names.index('trafficLight-Green'): new_names.index('trafficsignal'), # 4 -> 3
    original_names.index('trafficLight-GreenLeft'): new_names.index('trafficsignal'), # 5 -> 3
    original_names.index('trafficLight-Red'): new_names.index('trafficsignal'),     # 6 -> 3
    original_names.index('trafficLight-RedLeft'): new_names.index('trafficsignal'), # 7 -> 3
    original_names.index('trafficLight-Yellow'): new_names.index('trafficsignal'),# 8 -> 3
    original_names.index('trafficLight-YellowLeft'): new_names.index('trafficsignal'),# 9 -> 3
    original_names.index('truck'): new_names.index('car')                    # 10 -> 0
}

# --- EJECUCIÓN DEL SCRIPT ---

def remap_labels(directory):
    """
    Recorre un directorio de etiquetas y remapea los IDs de clase.
    """
    files_processed = 0
    labels_remapped = 0
                            
    if not os.path.exists(directory):
       print(f"  Advertencia: El directorio '{directory}' no existe. Omitiendo.")
       return 0, 0

    print(f" Procesando etiquetas en: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            temp_lines = []
                                                                                       
            with open(filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                                                                                                                                                                                                                                
                original_class_id = int(parts[0])                                                                                                                                                                                                                                
                if original_class_id in remapping:
                    new_class_id = remapping[original_class_id]
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n" 
                    temp_lines.append(new_line)
                    labels_remapped += 1

                else:
                    # Si por alguna razón hay un ID que no está en el mapa, lo mantenemos
                    temp_lines.append(line)

        # Escribir los cambios en el archivo
        with open(filepath, 'w') as f:
            f.writelines(temp_lines)
        
        files_processed += 1
    
    return files_processed, labels_remapped

# Itera sobre los directorios train, valid y test
total_files = 0
total_labels = 0

for split in ['train', 'valid', 'test']:
    labels_dir = os.path.join(dataset_dir, split, 'labels')
    files, labels = remap_labels(labels_dir)
    total_files += files
    total_labels += labels

print("\n ¡Remapeo completado!")
print(f"   - Archivos de etiquetas modificados: {total_files}")
print(f"   - Anotaciones totales remapeadas: {total_labels}")
