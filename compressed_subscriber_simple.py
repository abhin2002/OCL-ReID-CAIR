#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class CompressedSubscriber(Node):
    def __init__(self, topic='/zed/zed_node/left/image_rect_color/compressed'):
        super().__init__('compressed_subscriber_simple')
        self.bridge = CvBridge()
        self.get_logger().info(f'Subscribing to: {topic}')
        # simple QoS depth=10 (default reliability)
        self.sub = self.create_subscription(
            CompressedImage,
            topic,
            self.callback,
            10
        )
        cv2.namedWindow('remote_camera', cv2.WINDOW_NORMAL)

    def callback(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert CompressedImage: {e}')
            return

        cv2.imshow('remote_camera', cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit requested (q).')
            rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CompressedSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
