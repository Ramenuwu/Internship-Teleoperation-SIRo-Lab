#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

class TiagoEEPose(Node):

    def __init__(self):
        super().__init__('tiago_ee_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to query TF
        self.timer = self.create_timer(0.1, self.read_ee_pose)

        self.base_frame = 'base_footprint'
        self.ee_frame = 'arm_right_7_link'

    def read_ee_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                Time()
            )

            t = transform.transform.translation
            r = transform.transform.rotation

            self.get_logger().info(
                f"Position: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}] | "
                f"Quat: [{r.x:.3f}, {r.y:.3f}, {r.z:.3f}, {r.w:.3f}]"
            )

        except Exception as e:
            self.get_logger().warn(f"TF not available yet: {str(e)}")


def main():
    rclpy.init()
    node = TiagoEEPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()