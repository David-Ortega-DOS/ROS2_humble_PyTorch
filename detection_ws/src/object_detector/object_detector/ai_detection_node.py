import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesis
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D 
from ultralytics import YOLO
import torch

class AIDetectionNode(Node):
    def __init__(self):
        super().__init__('ai_detection_node')
        self.bridge = CvBridge()
                                    
        # 1. Cargar el modelo ENTRENADO (Reemplaza la ruta a tu archivo 'best.pt')
        model_path = '/home/david/yolo_custom_data/runs/detect/train3/weights/best.pt'
        self.model = YOLO(model_path)
                                                                    
        # 2. Configurar el dispositivo (GPU si está disponible)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Modelo cargado. Usando dispositivo: {self.device}')

        # 3. Suscripción a la imagen RGB del RPi 5
        # NOTA: Usamos QoS con historial de keep_last para baja latencia en la red.
        self.subscription = self.create_subscription( 
            Image,
            '/camera/color/image_raw', 
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data # Perfil optimizado para baja latencia (sensor data)
        )                                                                                                       
         
        # 4. Publicación de Detecciones 2D
        self.publisher_ = self.create_publisher(Detection2DArray, '/yolo/detections/two_d', 10)

    def image_callback(self, msg):
        try:
            # Convertir el mensaje ROS a una imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                                                          
        except Exception as e:
            self.get_logger().error(f"Error al convertir la imagen: {e}")
            return


        # --- 5. Inferencia de YOLOv8 ---
        # Pasamos la imagen, la resolución (imgsz=512) y el dispositivo ('cuda')
        results = self.model(
            cv_image, 
            imgsz=512, 
            device=self.device, 
            verbose=False, 
            stream=True, # Usa streaming para velocidad
            conf=0.3     # Umbral de confianza
        ) 
        
        detection_array = Detection2DArray()
        detection_array.header = msg.header # Esencial para la sincronización temporal (fusión 3D)

        for r in results:
            for box in r.boxes:
                # 6. Construir el mensaje Detection2D
                det = Detection2D()
                det.header.frame_id = msg.header.frame_id

                # Asignación de Centro
                det.bbox.center.x = float(box.xywh[0][0].item())
                det.bbox.center.y = float(box.xywh[0][1].item())
                det.bbox.center.theta = 0.0 # Inicializar theta a cero


                # Repite para el tamaño
                det.bbox.size_x = float(box.xywh[0][2].item())
                det.bbox.size_y = float(box.xywh[0][3].item())
                

                
                # Hipótesis (clase y confianza)
                hypothesis = ObjectHypothesis()
                hypothesis.class_id = self.model.names[int(box.cls[0])] 
                hypothesis.score = float(box.conf[0])
                det.results.append(hypothesis)
                
                detection_array.detections.append(det)
                        
        self.publisher_.publish(detection_array)


# Funciones main() quedan igual que en el ejemplo anterior
def main(args=None):
    rclpy.init(args=args)
    ai_detector = AIDetectionNode()
    rclpy.spin(ai_detector)
    ai_detector.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
