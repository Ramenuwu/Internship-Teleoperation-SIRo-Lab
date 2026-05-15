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
from sensor_msgs.msg import JointState


class TiagoImpedance(Node):

    def __init__(self):
        super().__init__("tiago_impedance")

        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")

        # Load URDF and create model/data
        urdf_path = "/intern_ws/tiago_mirror_moveit.urdf"
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Publisher for effort controller
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_right_effort_controller/commands',
            10
        )

        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.full_q = pin.neutral(self.model)
        self.full_dq = np.zeros(self.model.nv)
        
        # Subscriber
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10
        )
        
        self.joint_names = [
            "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
            "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
            "arm_right_7_joint",
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
        self.x_des = np.array([0.45, -0.5, 0.4]) #np.array([0.22436758, 0.27037221, 0.15008332])
        self.R_des = np.eye(3)
        self.x_curr = np.array([0,0,0])
        
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

        print("init done")

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
        if self.control_running:
            return
        self.control_running = True

        try:
            # ── 1. Analytic Jacobian via pinocchio (replaces _numeric_jacobian_pos) ──
            frame_id = self.model.getFrameId("arm_right_7_link")

            pin.forwardKinematics(self.model, self.data, self.full_q, self.full_dq, np.zeros(self.model.nv))   # zero ddq → data.a = J̇q̇
            pin.updateFramePlacements(self.model, self.data)

            J_full = pin.computeFrameJacobian(self.model, self.data, self.full_q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            J = J_full[0:3, :]                              # positional rows, (3 × nv)
            J_arm = J[:, self.arm_vel_idx]                  # (3 × 7)

            #print(J_arm)

            # Current end-effector position
            self.x_curr = self.data.oMf[frame_id].translation.copy()

            # J̇q̇  — bias acceleration (Coriolis in Cartesian space)
            #   pinocchio gives this directly when ddq=0 was passed above
            frame_acc   = pin.getFrameAcceleration(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jdot_qdot   = frame_acc.linear                  # (3,)

            # ── 2. Cartesian mass matrix  Λ(q) = (J M⁻¹ Jᵀ)⁻¹  (eq. 12) ──────────
            M_full = pin.crba(self.model, self.data, self.full_q)
            M_arm  = M_full[np.ix_(self.arm_vel_idx, self.arm_vel_idx)]  # (7 × 7)

            lambda_damp  = 1e-6
            JMinvJt      = J_arm @ np.linalg.solve(M_arm, J_arm.T)
            Lambda       = np.linalg.inv(JMinvJt + lambda_damp * np.eye(3))  # (3 × 3)

            # ── 3. Impedance gains ───────────────────────────────────────────────────
            #   Tune Kp first; Kd is derived for critical-ish damping (paper Sec. VI A)
            #   Dd = A·Kd1 + Kd1·A  where A²=Λ, Kd1²=Kp  → diagonal approx below
            Kp = np.diag([1000.0, 1000.0, 1000.0])
            xi = 0.1   # damping ratio per axis (1.0 = critically damped)

            # Factorization damping design (paper eq. 27):
            #   Dd = 2 · ξ · sqrt(Λ) · sqrt(Kp)
            Lambda_sqrt = np.diag(np.sqrt(np.diag(Lambda)))   # diagonal approximation
            Kp_sqrt     = np.diag(np.sqrt(np.diag(Kp)))
            Kd = 2.0 * xi * (Lambda_sqrt @ Kp_sqrt)           # (3 × 3), config-adaptive

            # ── 4. Errors ────────────────────────────────────────────────────────────
            error  = self.x_curr - self.x_des          # position error  e_x

            v_des = 0.0
            v_ee   = J_arm @ self.dq                   # end-effector velocity
            v_err  = v_ee - v_des                       # velocity error  ė_x  (v_des=0)

            # ── 5. Full task-space force law  (paper eq. 19) ─────────────────────────
            #
            #   F_τ = Λ(q)·ẍ_d  −  Kd·ė_x  −  Kp·e_x  −  Λ(q)·J̇(q)·q̇
            #         └──────┘    └────────────────────┘   └────────────┘
            #         ff accel    impedance forces          Coriolis corr.
            #
            xdd_des = np.zeros(3)   # constant target → zero desired acceleration

            F_tau   = (Lambda @ xdd_des
                       - Kd    @ v_err
                       - Kp    @ error
                       - Lambda @ Jdot_qdot)           # eq. 19

            tau_task = J_arm.T @ F_tau                # eq. 20 (NLE added below)

            # ── 6. Nullspace torques  N2 (paper eq. 17) ─────────────────────────────
            _, s_vals, Vt = np.linalg.svd(J_arm, full_matrices=True)
            rank    = int(np.sum(s_vals > 1e-6))
            Z       = Vt[rank:].T                      # (7 × 4) null-space basis
            N2      = M_arm @ (Z @ Z.T)                # dynamically consistent

            K_null  = 5
            D_null  = 3
            q_rest  = np.zeros(7)
            tau_d_N = (-K_null * (self.full_q[self.arm_q_idx] - q_rest)
                       - D_null * self.dq)
            tau_null = N2 @ tau_d_N

            # ── 7. Gravity + Coriolis  (eq. 20) ─────────────────────────────────────
            nle        = pin.nonLinearEffects(self.model, self.data, self.full_q, self.full_dq)
            nle_arm    = nle[self.arm_vel_idx]

            tau_des_arm = tau_null + tau_task + nle_arm * 0.465 # keep your empirical scale for now

            # ── 8. Publish ───────────────────────────────────────────────────────────
            self.effort_pub.publish(Float64MultiArray(data=tau_des_arm.tolist()))

            # debug topics
            self.pub_error.publish(Float64MultiArray(data=error.tolist()))
            self.pub_verr .publish(Float64MultiArray(data=v_err.tolist()))
            self.pub_fimp .publish(Float64MultiArray(data=F_tau.tolist()))
            self.pub_nle  .publish(Float64MultiArray(data=nle_arm.tolist()))
            self.pub_tau  .publish(Float64MultiArray(data=tau_des_arm.tolist()))
            self.pub_tau_task.publish(Float64MultiArray(data=tau_task.tolist()))
            self.pub_nle_full.publish(Float64MultiArray(data=nle.tolist()))
            self.pub_q_full  .publish(Float64MultiArray(data=self.full_q.tolist()))

        finally:
            self.control_running = False

    def _numeric_jacobian_pos(self, q, step=1e-3):
        """Compute Jacobian numerically"""
        n = q.shape[0]

        p0 = self._fk_pos(q)
        if p0 is None:
            print("FK failed!")
            return None, None


        J = np.zeros((3, n), dtype=float)

        for j in range(n):
            q_pert = q.copy()
            q_pert[j] += step

            p_pert = self._fk_pos(q_pert)
            if p_pert is None:
                self.get_logger().error(
                    f"FK failed while computing Jacobian column {j}"
                )
                return None, None

            J[:, j] = (p_pert - p0) / step

        return J, p0

    def _fk_pos(self, q):
        """Call FK service and wait for response"""
        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            print("Service not available!")
            return None
        
        req = GetPositionFK.Request()
        req.header.frame_id = "base_footprint"
        req.fk_link_names = ["arm_right_7_link"]
        
        rs = RobotState()
        rs.joint_state.name = self.joint_names
        rs.joint_state.position = q.tolist()
        req.robot_state = rs
        
        #print("Calling FK service...")
        future = self.fk_client.call_async(req)
        
        # Wait for response with proper ROS spinning
        count = 0
        while not future.done() and count < 100:
            rclpy.spin_once(self, timeout_sec=0.05)
            count += 1
            #print(count)
            
        if not future.done():
            print("Timed out waiting for FK response")
            return None
        
        res = future.result()

        if res is None:
            print("FK service call returned None")
            return None

        if res.error_code.val != res.error_code.SUCCESS:
            print(f"FK FAILED with code {res.error_code.val}")
            return None
        
        pose: PoseStamped = res.pose_stamped[0]
        p = pose.pose.position
        return np.array([p.x, p.y, p.z], dtype=float)


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


def step(node, eePos=None):
    """Update desired pose from WebSocket"""
    if eePos is not None:
        node.x_des = np.array([eePos[0], eePos[1], eePos[2]])
        quat = pin.Quaternion(eePos[3], eePos[4], eePos[5], eePos[6])
        quat.normalize()
        node.R_des = quat.toRotationMatrix()
        print("newpos")


def websocket_handler(websocket, node):
    """Handle incoming WebSocket messages"""
    for message in websocket:
        try:
            data = json.loads(message)
            print(f"Received: {data}")
            step(node, data)
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
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()





'''# Check 2: cross-check pinocchio FK against your MoveIt FK service
            #   at the same q, both should return the same xyz
            pin.forwardKinematics(self.model, self.data, self.full_q)
            pin.updateFramePlacements(self.model, self.data)
            frame_id = self.model.getFrameId("arm_right_7_link")
            
            x_pinocchio = self.data.oMf[frame_id].translation
            x_moveit    = self._fk_pos(self.q)   # your existing service call
            
            print("pinocchio:", x_pinocchio)
            print("moveit:   ", x_moveit)
            print("diff:     ", np.linalg.norm(x_pinocchio - x_moveit))

'''