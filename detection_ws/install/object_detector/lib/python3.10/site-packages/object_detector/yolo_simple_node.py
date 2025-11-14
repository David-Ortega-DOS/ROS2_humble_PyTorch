import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch

# El TOPIC DE SALIDA cambia a un array de números flotantes
YOLO_OUTPUT_TOPIC = '/yolo/detections/raw'
IMAGE_INPUT_TOPIC = '/camera/color/image_raw'
MODEL_PATH = '/home/david/yolo_custom_data/runs/detect/train3/weights/best.pt'

class YOLOSimpleNode(Node):
    def __init__(self):
        super().__init__('yolo_simple_node')
        self.bridge = CvBridge()
                                
        # 1. Cargar el modelo YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'Modelo YOLOv8 cargado. Dispositivo: {self.device}')

        # 2. Suscripción a la imagen RGB
        self.subscription = self.create_subscription( 
            Image,
            IMAGE_INPUT_TOPIC, 
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data
        )
          
        # 3. Publicación de Detecciones Crudas (Float32MultiArray)
        # Cada detección será una lista plana: [x_center, y_center, width, height, conf_score, class_id]
        self.publisher_ = self.create_publisher(Float32MultiArray, YOLO_OUTPUT_TOPIC, 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error CV Bridge: {e}")
            return

        # --- 4. Inferencia de YOLOv8 ---
        results = self.model(
            cv_image, 
            imgsz=512,  # Usaremos 512, puedes cambiar a 640/312
            device=self.device, 
            verbose=False, 
            stream=True, 
            conf=0.3    # Umbral de confianza
        ) 
        
        output_array = Float32MultiArray()
        detection_data = []

        for r in results:
            if r.boxes:
                for box in r.boxes:
                    # Coordenadas y Dimensiones (PyTorch tensor a float)
                    x_center = box.xywh[0][0].item() 
                    y_center = box.xywh[0][1].item()
                    width = box.xywh[0][2].item()
                    height = box.xywh[0][3].item()
                    
                    # Confianza y Clase
                    conf_score = box.conf[0].item()
                    class_id = float(box.cls[0].item()) # Aseguramos que sea float
                    
                    # 5. Llenar el array de datos
                    # Estructura: [x_center, y_center, width, height, conf_score, class_id]
                    detection_data.extend([x_center, y_center, width, height, conf_score, class_id])

        
        if detection_data:
            output_array.data = detection_data
            self.publisher_.publish(output_array)
            # self.get_logger().info(f"Detectado {len(detection_data)//6} objetos.")
        
        # Publicar un array vacío si no hay detecciones (para mantener el flujo de datos)
        else:
            self.publisher_.publish(output_array)

def main(args=None):
    rclpy.init(args=args)
    
    # ⚠️ Es importante envolver el nodo en un try-except para que el proceso no muera
    try:
        yolo_node = YOLOSimpleNode()
        rclpy.spin(yolo_node)
    except Exception as e:
        rclpy.logging.get_logger('yolo_simple_node').error(f"Error fatal: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()