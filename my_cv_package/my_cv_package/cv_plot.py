import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class CompressedImageSubscriber(Node):
    def __init__(self):
        super().__init__('compressed_image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )
        self.subscription  # to prevent unused variable warning

    def listener_callback(self, msg):
        # Convertir les données compressées en tableau numpy
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Décoder l'image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is not None:
            cv2.imshow("Compressed Image", image)
            cv2.waitKey(1)
        else:
            self.get_logger().warn("Failed to decode compressed image")

def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()