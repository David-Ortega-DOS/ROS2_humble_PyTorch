import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
# Nota: Eliminamos Marker, MarkerArray, ColorRGBA, Point, ya que solo usaremos la imagen

# --- CONFIGURACIÓN DE PARÁMETROS ---
IMAGE_INPUT_TOPIC = '/camera/color/image_raw'
IMAGE_OUTPUT_TOPIC = '/yolo/image_out' # El tópico que verás en Rviz2
MODEL_PATH = '/home/david/yolo_custom_data/runs/detect/train3/weights/best.pt'

class AIDetectionNode(Node): # Renombrado el nodo a AIDetectionNode
    def __init__(self):
        super().__init__('ai_detection_node')
        self.bridge = CvBridge()
        
        # 1. Cargar el modelo YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'Modelo YOLOv8 cargado. Dispositivo: {self.device}')

        # 2. Suscripción a la imagen RGB
        self.create_subscription( 
            Image,
            IMAGE_INPUT_TOPIC, 
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data
        )
          
        # 3. Publicación de la IMAGEN PROCESADA (¡La clave de la visualización!)
        self.image_pub = self.create_publisher(Image, IMAGE_OUTPUT_TOPIC, 10) 
        
        self.frame_count = 0
        self.frame_skip = 3 # Procesar solo 1 de cada 3 frames (aprox 10 FPS si la cámara es de 30 FPS)

    def image_callback(self, msg):
        # 1. SALTAR FRAMES
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
        # 2. Reiniciar el contador
        self.frame_count = 0

        try:
            # Convertir el mensaje ROS a una imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error CV Bridge: {e}")
            return

        # --- 4. Inferencia de YOLOv8 ---
        results = self.model(cv_image, imgsz=512, device=self.device, verbose=False, stream=True, conf=0.7) 
        
        # Eliminamos marker_array y marker_id, ya no se usan.

        for r in results:
            if r.boxes:
                for box in r.boxes:
                    
                    # Obtener la caja en formato xmin, ymin, xmax, ymax (formato píxel)
                    # Ultralytics proporciona xyxy
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) 
                    
                    # Clase y Confianza
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id]
                    conf_score = box.conf[0].item()

                    # ----------------------------------------------------
                    # 1. Elegir Color BGR para OpenCV
                    # BGR: (Blue, Green, Red)
                    color = (0, 255, 0) # Verde por defecto
                    if class_name == 'car':
                        color = (255, 0, 0) # Azul
                    elif class_name == 'person':
                        color = (0, 0, 255) # Rojo
                    
                    # 2. Dibujar el recuadro (Rectangle)
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 3. Preparar la etiqueta
                    label = f'{class_name} {conf_score:.2f}'
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    # 4. Fondo del texto
                    cv2.rectangle(cv_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    
                    # 5. Texto
                    cv2.putText(cv_image, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    # ----------------------------------------------------
        
        # *** Publicación Final de la Imagen Procesada ***
        try:
            # Convertir la imagen de OpenCV (con los recuadros) de nuevo a un mensaje ROS
            ros_output_img = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            ros_output_img.header = msg.header # Esencial: Usar el timestamp original
            self.image_pub.publish(ros_output_img)
        except Exception as e:
            self.get_logger().error(f"Error al publicar imagen: {e}")
        
        # Eliminamos self.publisher_.publish(marker_array)
        

def main(args=None):
    rclpy.init(args=args)
    try:
        ai_detector = AIDetectionNode()
        rclpy.spin(ai_detector)
    except Exception as e:
        rclpy.logging.get_logger('ai_detection_node').error(f"Error fatal: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()