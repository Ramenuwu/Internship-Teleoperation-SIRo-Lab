import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class GravComp(Node):
    def __init__(self):
        super().__init__("right_gravcomp")

        self.effort_right_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_right_effort_controller/commands',
            10
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm_right_effort_controller/torque_commands",
            self.additional_effort_cb,
            10
        )

        self.joint_names = [
            "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
            "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
            "arm_right_7_joint",
        ]

        urdf_path = "src/teleop/tiago_mirror_moveit.urdf"
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.full_q = pin.neutral(self.model)

        self.arm_vel_idx = [
            self.model.joints[self.model.getJointId(name)].idx_v
            for name in self.joint_names
        ]

        self.additional_efforts = np.zeros(len(self.joint_names))

    def additional_effort_cb(self, msg: Float64MultiArray):

        efforts = np.array(msg.data)

        #safety for wrong usage
        if len(efforts) != len(self.joint_names):
            self.get_logger().warn(
                f"Expected {len(self.joint_names)} efforts, got {len(efforts)}. Ignoring."
            )
            return
        
        self.additional_efforts = efforts

    def joint_state_cb(self, msg: JointState):
        # Update the full q vector from joint states
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for name in msg.name:
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                self.full_q[self.model.joints[jid].idx_q] = msg.position[name_to_index[name]]

        # Gravity compensation + any additional efforts from other nodes
        gravity_full = pin.computeGeneralizedGravity(self.model, self.data, self.full_q)
        gravcomp_tau_arm = gravity_full[self.arm_vel_idx] * 0.465
        total_tau = gravcomp_tau_arm + self.additional_efforts

        #print(total_tau)
        self.effort_right_pub.publish(Float64MultiArray(data=total_tau.tolist()))

def main():
    rclpy.init()
    node = GravComp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()