import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32MultiArray
import message_filters
import ros2_numpy as rnp
import numpy as np

# --- CONFIGURACIÓN DE PARÁMETROS ---
TARGET_FRAME = 'camera_link' 
QUEUE_SIZE = 10
TIME_SLOP = 0.1 
DEPTH_TOPIC = '/camera/depth/points'
YOLO_TOPIC = '/yolo/detections/raw'

# Mapeo de IDs de clase a nombres (debe coincidir con tu modelo YOLO)
# Basado en tu log donde class_id era 2.0
CLASS_MAPPING = {
    0: 'person', 
    1: 'car', 
    2: 'traffic_sign', # Asumimos que 2.0 es traffic_sign por el ejemplo, ajústalo a tu modelo.
    3: 'truck',
    # ... añade más clases según tu entrenamiento
}

# Colores para las visualizaciones de Rviz2 (R, G, B, A=Alpha)
COLOR_MAP = {
    'car': ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),       # Azul
    'person': ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),    # Rojo
    'traffic_sign': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8), # Verde
}


class DepthFusionNode(Node):
    def __init__(self):
        super().__init__('depth_fusion_node')

        # 1. Suscriptores sincronizados (Cambiado de Detection2DArray a Float32MultiArray)
        self.det_sub = message_filters.Subscriber(self, Float32MultiArray, YOLO_TOPIC)
        self.points_sub = message_filters.Subscriber(self, PointCloud2, DEPTH_TOPIC)

        # 2. Sincronizador Temporal 
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.det_sub, self.points_sub], 
            queue_size=QUEUE_SIZE, 
            slop=TIME_SLOP,
            allow_headerless=True 
        )
        self.ts.registerCallback(self.fusion_callback)

        # 3. Publicadores de resultados
        self.marker_pub = self.create_publisher(MarkerArray, '/detections/markers', 10)
        
        # 4. Nota: Detection3DArray no se implementará en este código simplificado
        # self.detection3d_pub = self.create_publisher(Detection3DArray, '/detections/tres_d', 10)
        
        self.get_logger().info(f'Nodo de Fusión 3D iniciado. Sincronizando: {YOLO_TOPIC} y {DEPTH_TOPIC}')

    def fusion_callback(self, raw_det_msg, points_msg):
        
        # Si el mensaje de YOLO está vacío, salimos y publicamos un array vacío
        if not raw_det_msg.data:
            self.marker_pub.publish(MarkerArray())
            return

        # 1. Desempaquetar el Float32MultiArray de YOLO
        # El array es plano: [x_center, y_center, w, h, conf, class_id] repetido
        try:
            raw_detections = np.array(raw_det_msg.data, dtype=np.float32).reshape(-1, 6)
        except ValueError:
            self.get_logger().warn("Array de detección con tamaño incorrecto. Ignorando frame.")
            return

        # Convertir la Nube de Puntos a un array de NumPy (N x 3 -> X, Y, Z)
        try:
            # 1. Usar numpify (devuelve un array estructurado N x 1)
            points_data_structured = rnp.numpify(points_msg)
            
            # CORRECCIÓN CLAVE 1: Verificar el tamaño usando la clave de la estructura
            if points_data_structured.size == 0:
                self.get_logger().warn("Nube de puntos vacía.")
                return

            # 2. Extraer las coordenadas XYZ como una matriz N x 3
            # Extraemos los campos 'x', 'y', 'z' del array estructurado
            points_data_xyz = np.stack([
                points_data_structured['x'],
                points_data_structured['y'],
                points_data_structured['z']
            ], axis=-1)

            # CORRECCIÓN CLAVE 2: Remodelar a 2D (alto x ancho x 3)
            # Esta matriz 2D es la que usaremos para indexar.
            points_array_2d = points_data_xyz.reshape(points_msg.height, points_msg.width, 3)

        except Exception as e:
            self.get_logger().error(f"Error al procesar PointCloud2: {e}")
            return

        marker_array = MarkerArray()

        for i, det_row in enumerate(raw_detections):
            # 2. Extracción de datos (índices 0, 1, 2, 3, 5)
            x_center_raw = det_row[0]
            y_center_raw = det_row[1]
            width = det_row[2]
            height = det_row[3]
            class_id = int(det_row[5])
            
            class_name = CLASS_MAPPING.get(class_id, 'unknown')
            
            # Convertir coordenadas del centro a enteros para indexar la matriz 2D
            x_center = int(x_center_raw)
            y_center = int(y_center_raw)
            
            # --- 3. Cálculo de la Posición y Distancia (Mediana de Z) ---
            
            # Usamos un parche central para mayor robustez (5x5 pixeles)
            patch_size = 2 # El parche será de 5x5 (2 alrededor del centro)
            
            # Definir límites del parche (asegurar que no salga de la imagen)
            x_min = max(0, x_center - patch_size)
            x_max = min(points_msg.width, x_center + patch_size + 1) # +1 para incluir el borde
            y_min = max(0, y_center - patch_size)
            y_max = min(points_msg.height, y_center + patch_size + 1)
            
            # Extraer la región del parche (solo coordenadas X, Y, Z)
            patch = points_array_2d[y_min:y_max, x_min:x_max, :]
            
            # Filtrar valores inválidos (NaNs o inf) y obtener las coordenadas Z válidas
            valid_x = patch[:, :, 0].flatten()[np.isfinite(patch[:, :, 0].flatten())]
            valid_y = patch[:, :, 1].flatten()[np.isfinite(patch[:, :, 1].flatten())]
            valid_z = patch[:, :, 2].flatten()[np.isfinite(patch[:, :, 2].flatten())]
            
            if valid_z.size < 5: # Requiere al menos 5 puntos válidos en el parche
                continue
            
            # Usamos la mediana de X, Y, Z para ser robustos a outliers
            center_x = np.median(valid_x)
            center_y = np.median(valid_y)
            distance_z = np.median(valid_z)
            
            # Si la distancia Z es menor a 0.1m o mayor a 10m, la descartamos
            if distance_z < 0.1 or distance_z > 10.0:
                continue

            # --- 4. Construcción del mensaje MarkerArray ---
            marker = Marker()
            marker.header.frame_id = points_msg.header.frame_id # Usamos el frame_id de la nube (ej: camera_depth_optical_frame)
            marker.header.stamp = points_msg.header.stamp
            marker.ns = class_name
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Posición 3D calculada
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = distance_z
            
            # Orientación (Identidad)
            marker.pose.orientation.w = 1.0 
            
            # Escala (dimensiones físicas estimadas)
            if class_name == 'person':
                marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.5, 1.8 
            elif class_name == 'car':
                marker.scale.x, marker.scale.y, marker.scale.z = 2.0, 1.5, 1.5
            else: # Señal de tráfico / otros
                marker.scale.x, marker.scale.y, marker.scale.z = 0.2, 0.05, 0.8
            
            # Color
            marker.color = COLOR_MAP.get(class_name, ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8))
            
            marker_array.markers.append(marker)
            
        # Publicar los resultados de MarkerArray
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        depth_fusion = DepthFusionNode()
        rclpy.spin(depth_fusion)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rclpy.logging.get_logger('depth_fusion_node').error(f"Error fatal: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()