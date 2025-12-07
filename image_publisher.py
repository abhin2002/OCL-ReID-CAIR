#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import time
import numpy as np

# 
class ImagePublisher(Node):
    def __init__(self, camera_index=0, fps=30.0):
        super().__init__('image_publisher')
        # Publish CompressedImage instead of custom ImageFrame
        self.publisher_ = self.create_publisher(CompressedImage, '/zed/zed_node/left/image_rect_color/compressed', 10)
        # self.publisher_ = self.create_publisher(CompressedImage, 'image_topic', 10)
        self.bridge = CvBridge()

        # --- Open the webcam ---
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'Unable to open camera index {camera_index}')
            raise RuntimeError(f'Unable to open camera index {camera_index}')

        # Set camera parameters (best-effort)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, float(fps))

        # --- Warm up the camera ---
        self.get_logger().info("[INFO] Warming up the webcam...")
        time.sleep(2.0)  # Give hardware time to start
        self._wait_for_valid_frame()

        self.timer_period = 1.0 / float(fps)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.counter = 0

    def _wait_for_valid_frame(self, max_attempts=50):
        """Try to get a valid, non-black frame from the webcam before starting."""
        attempt = 0
        while attempt < max_attempts:
            ret, frame = self.cap.read()
            if ret and np.sum(frame) > 1000:  # not completely black
                self.get_logger().info(f"[INFO] Camera ready after {attempt + 1} attempts.")
                return True
            attempt += 1
            time.sleep(0.1)
        self.get_logger().warning("[WARNING] Could not get a well-lit frame. Continuing anyway.")
        return False

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning('Failed to capture frame from camera')
            return

        # Convert BGR (OpenCV) -> ROS CompressedImage (JPEG)
        try:
            comp_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
            comp_msg.header.stamp = self.get_clock().now().to_msg()
            comp_msg.header.frame_id = 'camera_frame'
        except Exception as e:
            self.get_logger().warning(f'Failed to convert frame to CompressedImage: {e}')
            return

        self.publisher_.publish(comp_msg)
        self.get_logger().info(f'Publishing compressed image {self.counter}')
        self.counter += 1

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
                self.get_logger().info('Camera released.')
        except Exception as e:
            self.get_logger().warning(f'Error releasing camera: {e}')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImagePublisher(camera_index=0, fps=15.0)
        rclpy.spin(node)
    except Exception as e:
        print(f'Exception in image_publisher: {e}')
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
