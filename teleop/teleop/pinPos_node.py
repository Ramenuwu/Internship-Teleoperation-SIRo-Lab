import json
import threading
import time
from websockets.sync.server import serve

import pinocchio as pin

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from moveit_msgs.srv import GetPositionFK
import numpy as np
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState


class TiagoImpedance(Node):

    def __init__(self):
        super().__init__("tiago_impedance")

        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")

        # Load URDF and create model/data
        urdf_path = "src/teleop/tiago_mirror_moveit.urdf"
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Publisher for effort controller
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_left_effort_controller/torque_commands',
            10
        )

        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.full_q = pin.neutral(self.model)
        self.full_dq = np.zeros(self.model.nv)

        self.EEweightkg = 0.07768 #kg
        self.EEweightN = self.EEweightkg * 9.81
        
        # Subscriber
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10
        )

        self.create_subscription(
            WrenchStamped,
            "/ft_sensor_left_controller/wrench",
            self.wrench_cb,
            10
        )
        
        self.joint_names = [
            "arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint",
            "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint",
            "arm_left_7_joint",
        ]

        #wheels are NAN in joint_states
        self.ignore_joints = {
            "wheel_front_left_joint", "wheel_front_right_joint",
            "wheel_rear_left_joint", "wheel_rear_right_joint"
        }
        
        self.arm_q_idx = [
            self.model.joints[self.model.getJointId(name)].idx_q
            for name in self.joint_names
        ]
        self.arm_vel_idx = [
            self.model.joints[self.model.getJointId(name)].idx_v
            for name in self.joint_names
        ]

        # Control state
        self.base_offset = np.array([0.0, 0.2, 0.7])

        self.x_des = np.array([0.0, 0.0, 0.0]) #0.279, -0.209, 0.0
        self.R_des = np.diag([1.0,1.0,1.0])
        self.x_curr = np.array([0,0,0])

        self.F_ext = np.zeros(6)

        self.F_des = np.zeros(6)
        self.F_tau = np.zeros(6)

        self.alpha = 0.2
        
        # Flag to prevent concurrent control calculations
        self.control_running = False

        #plottin
        self.pub_error    = self.create_publisher(Float64MultiArray, '/dbg/error',       10)
        self.pub_verr     = self.create_publisher(Float64MultiArray, '/dbg/v_err',       10)
        self.pub_fimp     = self.create_publisher(Float64MultiArray, '/dbg/f_imp',       10)
        self.pub_nle      = self.create_publisher(Float64MultiArray, '/dbg/nle_arm',     10)
        self.pub_tau      = self.create_publisher(Float64MultiArray, '/dbg/tau_des_arm', 10)
        self.pub_tau_task      = self.create_publisher(Float64MultiArray, '/dbg/tau_task_arm', 10)
        self.pub_nle_full      = self.create_publisher(Float64MultiArray, '/dbg/nle_full', 10)
        self.pub_q_full      = self.create_publisher(Float64MultiArray, '/dbg/q_full', 10)
        self.pub_F_ext      = self.create_publisher(Float64MultiArray, '/dbg/F_ext', 10)

        print("init done")

    def wrench_cb(self, msg: WrenchStamped):
        
        R = self.data.oMf[self.model.getFrameId("arm_left_7_link")].rotation

        f_sensor = np.array([msg.wrench.force.x,
                             msg.wrench.force.y,
                             msg.wrench.force.z])
        t_sensor = np.array([msg.wrench.torque.x,
                             msg.wrench.torque.y,
                             msg.wrench.torque.z])
        
        # Rotate from sensor frame into world/base_footprint frame
        self.F_ext[0:3] = R @ f_sensor
        self.F_ext[3:6] = R @ t_sensor
        #compensate sensor weight: 0.6736N straight downwards
        self.F_ext[2] += self.EEweightN
        #self.F_ext = np.zeros(6)

        if self.data.oMf[self.model.getFrameId("arm_left_7_link")].translation[0] > 0.25:      # places a virtual wall at x 3.5
            self.F_ext[0] -= self.F_tau[0]
            print("hittin wall")


    def joint_state_cb(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for name in msg.name:
            if name in self.ignore_joints:
                continue
            if not self.model.existJointName(name):
                continue
            
            jid = self.model.getJointId(name)
            idx_msg = name_to_index[name]

            self.full_q[self.model.joints[jid].idx_q] = msg.position[idx_msg]
            self.full_dq[self.model.joints[jid].idx_v] = msg.velocity[idx_msg]

            if name in self.joint_names:
                i = self.joint_names.index(name)
                self.q[i] = msg.position[idx_msg]
                self.dq[i] = msg.velocity[idx_msg]

    def control_timer_callback(self):
        #if self.control_running:
        #    return
        #self.control_running = True

        try:
            frame_id = self.model.getFrameId("arm_left_7_link")

            # ── 1. Kinematics ────────────────────────────────────────────────────────
            pin.forwardKinematics(self.model, self.data, self.full_q, self.full_dq, np.zeros(self.model.nv))
            pin.updateFramePlacements(self.model, self.data)

            # Full 6×nv Jacobian (rows 0:3 = linear, 3:6 = angular)
            J_full = pin.computeFrameJacobian(self.model, self.data, self.full_q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J6_arm = J_full[:, self.arm_vel_idx]        # (6 × 7)

            # Current pose
            T_curr      = self.data.oMf[frame_id]
            self.x_curr = T_curr.translation.copy()     # (3,)
            R_curr      = T_curr.rotation               # (3 × 3)

            # J̇q̇  — full 6D bias acceleration
            frame_acc = pin.getFrameAcceleration(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jdot_qdot_6 = np.concatenate([frame_acc.linear, frame_acc.angular])               # (6,)

            # ── 2. Cartesian mass matrix Λ(q) ─────────────────────────────
            M_full  = pin.crba(self.model, self.data, self.full_q)
            M_arm   = M_full[np.ix_(self.arm_vel_idx, self.arm_vel_idx)]   # (7 × 7)

            JMinvJt = J6_arm @ np.linalg.solve(M_arm, J6_arm.T)
            Lambda  = np.linalg.inv(JMinvJt + 1e-6 * np.eye(6))           # (6 × 6)

            # ── 3. Gains ─────────────────────────────────────────────────────────────────
            # Desired inertia. can be set freely, independent of Λ(q), i think its not actually supposed to be changed tho.
            # Start with Λ_d = Λ(q) (identity scaling) so behaviour matches eq 19,
            # then reduce to make arm feel lighter to the environment
            Lambda_d = Lambda.copy()    # (6×6)
            Lambda_d_inv = np.linalg.inv(Lambda_d)

            Kp = np.diag([1000.0, 1000.0, 1000.0,
                        50.0,  50.0,  50.0])

            xi = 0.5
            Lambda_d_sqrt = np.diag(np.sqrt(np.diag(Lambda_d)))
            Kp_sqrt       = np.diag(np.sqrt(np.diag(Kp)))

            # Damping designed against Λ_d now, not Λ(q)
            Kd = 2.0 * xi * (Lambda_d_sqrt @ Kp_sqrt)              # (6×6)

            # ── 4. Errors ─────────────────────────────────────────────────────────────────
            e_pos   = self.x_curr - (self.x_des + self.base_offset)
            R_err   = R_curr @ self.R_des.T
            e_ori   = pin.log3(R_err)
            error_6 = np.concatenate([e_pos, e_ori])                # (6,)

            v_6     = J6_arm @ self.dq                              # (6,)

            # ── 5. Full impedance law ──────────────────────────────────────────
            #
            #   F_τ = Λ(q)·ẍ_d
            #         − Λ(q)·Λ_d⁻¹·(Dd·ė_x + Kd·e_x)    ← stiffness/damping scaled by Λ/Λ_d
            #         + (Λ(q)·Λ_d⁻¹ − I)·F_ext           ← inertia shaping from F_ext
            #         − Λ(q)·J̇q̇                           ← Coriolis correction
            #
            xdd_des     = np.zeros(6)

            LLdinv      = Lambda @ Lambda_d_inv                     # (6×6)

            self.F_tau = (Lambda  @ xdd_des
                     - LLdinv @ ((Kd @ v_6 + Kp @ error_6) * self.alpha)
                     + (LLdinv - np.eye(6)) @ self.F_ext
                     + self.F_des
                     - Lambda @ Jdot_qdot_6)                    # (6,)
  
            
            tau_task = J6_arm.T @ self.F_tau                     # (7,)

            # ── 6. Nullspace torques N2 ───────────────────────────────
            _, s_vals, Vt = np.linalg.svd(J6_arm, full_matrices=False)
            rank     = int(np.sum(s_vals > 1e-6))    # should be 6 for non-singular
            Z        = Vt[rank:].T                    # (7 × 1) for a 7-DOF arm
            N2       = M_arm @ (Z @ Z.T)

            q_min  = np.array([-0.5235987755982988,-2.443460952792061,-2.6179938779914944,-2.443460952792061,-3.6651914291880923,-1.8849555921538759,-2.6179938779914944])   # your robot's lower limits
            q_max  = np.array([4.71238898038469,1.1344640137963142,2.6179938779914944,1.1344640137963142,1.5707963267948966,3.001966313430247,2.6179938779914944])   # your robot's upper limits
            q_mid  = 0.5 * (q_min + q_max)


            K_null   = 5.0
            D_null   = 3.0
            tau_d_N = -K_null * (self.q - q_mid) - D_null * self.dq
            tau_null = N2 @ tau_d_N

            # ── 7. Gravity + Coriolis ────────────────────────────────────────────────
            nle        = pin.nonLinearEffects(self.model, self.data, self.full_q, self.full_dq)
            grav       = pin.computeGeneralizedGravity(self.model, self.data, self.full_q)
            nle_nograv = nle - grav

            #nle_grav_scaled = nle_nograv + grav*0.465          #Gravity not needed with the new custom Controller

            #nle_arm    = nle_grav_scaled[self.arm_vel_idx]
            nle_arm    = nle_nograv[self.arm_vel_idx]

            tau_des_arm_calc = tau_task + tau_null + nle_arm

            tau_des_arm = np.clip(tau_des_arm_calc, -15, 15) ##clip to 15N

            # ── 8. Publish ───────────────────────────────────────────────────────────
            self.effort_pub.publish(Float64MultiArray(data=tau_des_arm.tolist()))

            self.pub_error   .publish(Float64MultiArray(data=error_6.tolist()))
            self.pub_verr    .publish(Float64MultiArray(data=v_6.tolist()))
            self.pub_fimp    .publish(Float64MultiArray(data=self.F_tau.tolist()))
            self.pub_nle     .publish(Float64MultiArray(data=nle_arm.tolist()))
            self.pub_tau     .publish(Float64MultiArray(data=tau_des_arm.tolist()))
            self.pub_tau_task.publish(Float64MultiArray(data=tau_task.tolist()))
            self.pub_nle_full.publish(Float64MultiArray(data=nle.tolist()))
            self.pub_q_full  .publish(Float64MultiArray(data=self.full_q.tolist()))
            self.pub_F_ext       .publish(Float64MultiArray(data=self.F_ext.tolist()))

        finally:
            self.control_running = False

# -------- WEBSOCKET PART --------

def websocket_server_thread(node):
    """Run WebSocket server in a separate thread"""
    print("Starting WebSocket server on 192.168.0.201:8765")
    try:
        with serve(lambda ws: websocket_handler(ws, node), "192.168.0.201", 8765) as server:
            server.serve_forever()
            print("WebSocket server running")
    except Exception as e:
        print(f"WebSocket server error: {e}")


def step(node, statevec=None):
    """Update desired pose from WebSocket"""
    if statevec is not None:
        node.x_des = np.array([statevec[0], statevec[1], statevec[2]])
        quat = pin.Quaternion(statevec[3], statevec[4], statevec[5], statevec[6])
        quat.normalize()
        node.R_des = quat.toRotationMatrix()
        node.F_des = np.array([statevec[7], statevec[8], statevec[9], statevec[10], statevec[11], statevec[12]])
        print("newpos")


def websocket_handler(websocket, node):
    """Handle incoming WebSocket messages"""
    for message in websocket:
        try:
            data = json.loads(message)
            print(f"Received: {data}")
            step(node, data)

            response = json.dumps([node.F_ext[0],
                                   node.F_ext[1],
                                   node.F_ext[2],
                                   node.F_ext[3],
                                   node.F_ext[4],
                                   node.F_ext[5]]).encode()
            websocket.send(response)

            print("send:")
            print(response)

        except Exception as e:
            print(f"Error: {e}")


def main():
    rclpy.init()
    node = TiagoImpedance()

    #for i, frame in enumerate(node.model.frames):
    #    print(f"{i}: {frame.name}")

    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=websocket_server_thread, args=(node,), daemon=True)
    ws_thread.start()
    
    print("Waiting for WebSocket server to start...")
    time.sleep(1)

    print("Starting ROS spin...")
    
    # In main thread, periodically update current position
    # This runs concurrently with the control timer
    node.control_timer_callback()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            node.control_timer_callback()
    except KeyboardInterrupt:
        tau_zeros = np.array([0,0,0,0,0,0,0])
        node.effort_pub.publish(Float64MultiArray(data=tau_zeros.tolist()))
        print("Shutting down...")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()





'''# Check 2: cross-check pinocchio FK against your MoveIt FK service
            #   at the same q, both should return the same xyz
            pin.forwardKinematics(self.model, self.data, self.full_q)
            pin.updateFramePlacements(self.model, self.data)
            frame_id = self.model.getFrameId("arm_left_7_link")
            
            x_pinocchio = self.data.oMf[frame_id].translation
            x_moveit    = self._fk_pos(self.q)   # your existing service call
            
            print("pinocchio:", x_pinocchio)
            print("moveit:   ", x_moveit)
            print("diff:     ", np.linalg.norm(x_pinocchio - x_moveit))

'''