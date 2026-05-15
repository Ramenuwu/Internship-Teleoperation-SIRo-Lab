import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
import numpy as np

class FKTester(Node):
    def __init__(self):
        super().__init__("fk_tester")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        
    def test_fk(self):
        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            print("Service not available!")
            return
        
        req = GetPositionFK.Request()
        req.header.frame_id = "base_footprint"
        req.fk_link_names = ["arm_right_7_link"]
        
        rs = RobotState()
        rs.joint_state.name = [
            "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
            "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
            "arm_right_7_joint"
        ]
        rs.joint_state.position = [0.52360636, -2.44357124,  0.54957128, -2.44348232, -0.70235171, -0.435428, -2.45634967]  #[0.0] * 7
        req.robot_state = rs
        
        print("Calling FK service...")
        future = self.fk_client.call_async(req)
        
        # Wait for response
        count = 0
        while not future.done() and count < 50:
            rclpy.spin_once(self, timeout_sec=0.1)
            count += 1
            print(f"Waiting... {count}")
        
        if future.done():
            result = future.result()
            print(f"Success! Result: {result}")
        else:
            print("Timed out waiting for response")


def main():
    rclpy.init()
    node = FKTester()
    node.test_fk()
    rclpy.shutdown()