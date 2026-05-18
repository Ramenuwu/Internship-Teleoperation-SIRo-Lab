#!/usr/bin/env python3
import json
import asyncio
from websockets.asyncio.server import serve

import rclpy
from rclpy.node import Node

import numpy as np

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK


class TiagoIK(Node):

    def __init__(self):
        super().__init__("tiago_ik_node")

        self.joint_state = None

        # Subscriber
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10
        )

        # IK client
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

        # Publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/arm_right_controller/joint_trajectory",
            10
        )

        self.arm_joints = [
            "arm_right_1_joint",
            "arm_right_2_joint",
            "arm_right_3_joint",
            "arm_right_4_joint",
            "arm_right_5_joint",
            "arm_right_6_joint",
            "arm_right_7_joint",
        ]

        self.get_logger().info("Waiting for IK service...")
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass

        self.get_logger().info("Waiting for joint states...")
        while self.joint_state is None:
            rclpy.spin_once(self)

        self.get_logger().info("Ready")

        self.move_to_pose()

        # Timer for control loop
        #self.dt = 0.01
        #self.timer = self.create_timer(self.dt, self.move_to_pose) ##maybe this the problem?

    def joint_state_cb(self, msg):
        self.joint_state = msg

    def move_to_pose(self, eePos=None):
        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"

        '''
        pose.pose.position.x = eePos[0]#+0.1
        pose.pose.position.y = eePos[1]#-0.1
        pose.pose.position.z = eePos[2]#+0.6
        pose.pose.orientation.w = eePos[3]
        pose.pose.orientation.x = eePos[4]
        pose.pose.orientation.y = eePos[5]
        pose.pose.orientation.z = eePos[6]
        #'''

        #''' 0.8844352  -0.35999206 -0.46213063 the tiago said this is the position but ik fails with this
        pose.pose.position.x = 0.0
        pose.pose.position.y = -0.45
        pose.pose.position.z = 0.6
        pose.pose.orientation.w = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        #'''

        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm_right"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 1
        req.ik_request.robot_state.joint_state = self.joint_state

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()

        if res.error_code.val != 1:
            self.get_logger().error("IK failed")
            return

        names = res.solution.joint_state.name
        positions = res.solution.joint_state.position

        arm_positions = []
        for joint in self.arm_joints:
            idx = names.index(joint)
            arm_positions.append(positions[idx])

        traj = JointTrajectory()
        traj.joint_names = self.arm_joints

        point = JointTrajectoryPoint()
        point.positions = arm_positions
        point.time_from_start.sec = 2

        traj.points.append(point)

        self.traj_pub.publish(traj)
        self.get_logger().info(f"Trajectory sent for {eePos}")


# -------- WEBSOCKET PART --------

async def websocket_handler(websocket, node):
    async for message in websocket:
        try:
            data = json.loads(message)
            print(f"<<< {data}")

            node.move_to_pose(data)

        except Exception as e:
            print(f"Error: {e}")


async def ros_spin(node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        await asyncio.sleep(0.01)


async def main_async():
    rclpy.init()
    node = TiagoIK()

    '''
    server = await serve(
        lambda ws: websocket_handler(ws, node),
        "192.168.0.201",
        8765
    )

    print("WebSocket server started on port 8765")

    await asyncio.gather(
        ros_spin(node),
        server.serve_forever()
    )
    '''


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()