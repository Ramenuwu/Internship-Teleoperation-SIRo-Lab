#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ArmEffortPublisher(Node):
    def __init__(self, side='right', publish_period=0.1):
        super().__init__('arm_effort_publisher')
        self.side = side
        self.joints = [
            f'arm_{side}_1_joint',
            f'arm_{side}_2_joint',
            f'arm_{side}_3_joint',
            f'arm_{side}_4_joint',
            f'arm_{side}_5_joint',
            f'arm_{side}_6_joint',
            f'arm_{side}_7_joint',
        ]

        controller_name = f'arm_{side}_effort_controller'
        self.command_topic = f'/{controller_name}/commands'

        self.pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)

        self.default_effort = [0.0] * 7
        self.timer = self.create_timer(publish_period, self.timer_callback)

        self.get_logger().info(f'Publishing efforts on {self.command_topic}')

    def timer_callback(self):
        self.send_effort(self.default_effort)

    def send_effort(self, efforts):
        assert len(efforts) == 7, 'Effort vector must have 7 elements.'

        msg = Float64MultiArray()
        msg.data = list(efforts)
        self.pub.publish(msg)

        self.get_logger().debug(f'Sent effort: {efforts}')

    def set_effort(self, efforts):
        self.default_effort = list(efforts)


def main():
    rclpy.init()

    node = ArmEffortPublisher(side='right', publish_period=0.1)

    node.set_effort([0.0, +0.0, -0.0, +5.0, 0.0, -0.0, 0.0])

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
