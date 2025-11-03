#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PX4 (1.16) + uXRCE-DDS + ROS 2 (Humble) Offboard 速度控制环境封装（不使用 MAVROS）

功能：
- reset_and_hover(): 进入Offboard、解锁，起飞到指定高度悬停，然后切回速度模式等待速度命令
- step(action_str): 以 ENU 语义发送速度指令（内部自动映射到 PX4 NED）

依赖：
- rclpy
- px4_msgs
- （可选）sensor_msgs、cv_bridge（若缺失将只保留原始 Image 消息，不做数组转换）
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration

# PX4 ROS 2 消息
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

# 传感器（可选）
from sensor_msgs.msg import Image as RosImage

# 尝试引入 cv_bridge（若 Numpy/CvBridge 不兼容则自动降级）
try:
    from cv_bridge import CvBridge
    _HAS_CVBRIDGE = True
except Exception:
    CvBridge = None  # type: ignore
    _HAS_CVBRIDGE = False


def enu_vel_to_ned(vx_enu: float, vy_enu: float, vz_enu: float):
    """
    ENU(x East, y North, z Up)  →  NED(x North, y East, z Down)
    """
    vx_ned = vy_enu
    vy_ned = vx_enu
    vz_ned = -vz_enu
    return float(vx_ned), float(vy_ned), float(vz_ned)


def enu_pos_to_ned(x_enu: float, y_enu: float, z_enu: float):
    """
    ENU(x East, y North, z Up)  →  NED(x North, y East, z Down)
    """
    x_ned = y_enu
    y_ned = x_enu
    z_ned = -z_enu
    return float(x_ned), float(y_ned), float(z_ned)


def now_us(node: Node) -> int:
    """返回 ROS 2 时钟的微秒时间戳（PX4 消息要求微秒）"""
    return node.get_clock().now().nanoseconds // 1000


class Px4IsaacSimVelEnv(Node):
    """
    纯 ROS 2 + px4_msgs 环境：

    - 发布：
        /fmu/in/offboard_control_mode   (OffboardControlMode)
        /fmu/in/trajectory_setpoint     (TrajectorySetpoint)
        /fmu/in/vehicle_command         (VehicleCommand)
    - 订阅：
        /fmu/out/vehicle_local_position (VehicleLocalPosition)
        /fmu/out/vehicle_status         (VehicleStatus)
        （可选）/rgb, /depth （RosImage）

    - API:
        take_off(alt): 切入 Offboard 并可更新目标高度
        hover(position): 固定位置悬停
        vel_cnt(velocity): 速度模式控制
        step(velocity_enu, yaw_rate_deg=0.0): 以 ENU 速度指令控制
    """

    def __init__(self,
                 node_name: str = "px4_isaac_vel_env",
                 hover_alt: float = 1.0,
                 keepalive_hz: float = 20.0,
                 subscribe_rgb: bool = False,
                 subscribe_depth: bool = False):
        super().__init__(node_name)

        self.get_logger().info("PX4 IsaacSim Vel ENV started.")

        # ---------- QoS ----------
        # uXRCE-DDS / px4_msgs 常用 QoS：传感器数据用 BestEffort，控制用默认/可靠
        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_ctrl = 10  # 简单用整数 depth 的默认可靠 QoS

        # ---------- Publishers ----------
        self.pub_offb = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_ctrl)
        self.pub_ts   = self.create_publisher(TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",  qos_ctrl)
        self.pub_cmd  = self.create_publisher(VehicleCommand,      "/fmu/in/vehicle_command",      qos_ctrl)

        # ---------- Subscribers ----------
        self.sub_lpos = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_lpos, qos_sensor
        )
        self.sub_status = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, qos_sensor
        )

        self.rgb_img_np = None
        self.depth_img_np = None
        self.rgb_msg: Optional[RosImage] = None
        self.depth_msg: Optional[RosImage] = None
        self._bridge = CvBridge() if _HAS_CVBRIDGE else None

        if subscribe_rgb:
            self.create_subscription(RosImage, "/rgb", self._on_rgb, qos_sensor)
        if subscribe_depth:
            self.create_subscription(RosImage, "/depth", self._on_depth, qos_sensor)

        # ---------- State ----------
        self.hover_alt = float(hover_alt)          # 目标悬停高度（米）
        self.keepalive_period = 1.0 / float(keepalive_hz)

        self.armed = False
        self.nav_state = 0
        self.ned_x = math.nan
        self.ned_y = math.nan
        self.ned_z = math.nan  # VehicleLocalPosition.z（向下为正）
        self._ned_vx = math.nan
        self._ned_vy = math.nan
        self._ned_vz = math.nan
        self._heading_rad = math.nan

        # “最近一次 setpoint”，用于定时器保活重复发布
        self._last_sp: Optional[TrajectorySetpoint] = None
        self._use_velocity_mode = False  # 起飞阶段先用 position 模式

        # ---------- Timer 保活 ----------
        self._timer = self.create_timer(self.keepalive_period, self._on_timer)

    # ============================ Callbacks ============================

    def _on_lpos(self, msg: VehicleLocalPosition):
        self.ned_x = float(msg.x)
        self.ned_y = float(msg.y)
        self.ned_z = float(msg.z)  # NED：向下为正
        self._ned_vx = float(getattr(msg, "vx", math.nan))
        self._ned_vy = float(getattr(msg, "vy", math.nan))
        self._ned_vz = float(getattr(msg, "vz", math.nan))
        heading = getattr(msg, "heading", math.nan)
        if heading is not None:
            heading = float(heading)
            if math.isfinite(heading):
                self._heading_rad = heading
        print(
            "Local Position NED: "
            f"x={self.ned_x:.2f}, y={self.ned_y:.2f}, z={self.ned_z:.2f}, "
            f"vx={self._ned_vx:.2f}, vy={self._ned_vy:.2f}, vz={self._ned_vz:.2f}, "
            f"heading={heading}"
        )

    def _on_status(self, msg: VehicleStatus):
        # arming_state: VehicleStatus.ARMING_STATE_ARMED 等（整型枚举）
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.nav_state = msg.nav_state

    def _on_rgb(self, msg: RosImage):
        self.rgb_msg = msg
        if self._bridge is not None:
            try:
                import numpy as np
                self.rgb_img_np = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception:
                self.rgb_img_np = None  # 退化为只存消息

    def _on_depth(self, msg: RosImage):
        self.depth_msg = msg
        if self._bridge is not None:
            try:
                import numpy as np
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                # 简单约束：非正/NaN 用 max range 或 0 处理，按需调整
                img = np.nan_to_num(img, nan=10.0)
                img = img[:, :, None]
                self.depth_img_np = img
            except Exception:
                self.depth_img_np = None

    def _on_timer(self):
        """
        20Hz 定时器：持续发布 OffboardControlMode + 最近一次 TrajectorySetpoint（PX4 Offboard 需要保活）
        """
        # OffboardControlMode：根据当前阶段选择 position 或 velocity
        offb = OffboardControlMode()
        offb.timestamp = now_us(self)
        offb.position = not self._use_velocity_mode
        offb.velocity = self._use_velocity_mode
        offb.acceleration = False
        offb.attitude = False
        offb.body_rate = False
        offb.thrust_and_torque = False
        offb.direct_actuator = False
        self.pub_offb.publish(offb)

        # 复发最近一次 setpoint
        if self._last_sp is not None:
            # 刷新时间戳
            self._last_sp.timestamp = offb.timestamp
            self.pub_ts.publish(self._last_sp)

    # ============================ Core control ============================

    def _send_vehicle_command(self, command: int, **params):
        """
        发送通用 VehicleCommand
        常用：
          - ARM/DISARM: command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1/0
          - SET_MODE:   command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1(custom), param2=6(OFFBOARD)
        """
        cmd = VehicleCommand()
        cmd.timestamp = now_us(self)
        cmd.command = int(command)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True

        # 将 param1..param7 可选写入
        for i in range(1, 8):
            key = f"param{i}"
            if key in params:
                setattr(cmd, key, float(params[key]))

        self.pub_cmd.publish(cmd)


    def enter_offboard_and_arm(self):
        # 先 1.5s 保活（和你原来一致）
        t_end = self.get_clock().now() + Duration(seconds=0.05)
        while self.get_clock().now() < t_end:
            offb = OffboardControlMode()
            offb.timestamp = now_us(self)
            offb.position = True; offb.velocity = False
            self.pub_offb.publish(offb)

            sp = TrajectorySetpoint()
            sp.timestamp = offb.timestamp
            sp.position[:] = [0.0]*3; sp.velocity[:] = [math.nan]*3
            sp.acceleration[:] = [math.nan]*3
            if hasattr(sp, "jerk"): sp.jerk[:] = [math.nan]*3
            sp.yaw = math.nan; sp.yawspeed = 0.0
            self.pub_ts.publish(sp)
            rclpy.spin_once(self, timeout_sec=0.0)

        # 切 Offboard
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
    
        # Arm
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
        
        
    def take_off(self, hover_alt_m: Optional[float] = None):
        """
        进入 Offboard 并解锁，可选更新目标悬停高度。
        """
        if hover_alt_m is not None:
            self.hover_alt = float(hover_alt_m)

        self._use_velocity_mode = False  # 起飞阶段用 position
        self._last_sp = None
        while not self.armed or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.enter_offboard_and_arm()
            self.get_logger().info("Waiting for armed + offboard...")

        self.get_logger().info("Offboard + arm complete.")

    def _current_position_enu(self) -> Optional[tuple]:
        if math.isfinite(self.ned_x) and math.isfinite(self.ned_y) and math.isfinite(self.ned_z):
            return float(self.ned_y), float(self.ned_x), float(-self.ned_z)
        return None

    def _current_speed(self) -> float:
        if math.isfinite(self._ned_vx) and math.isfinite(self._ned_vy) and math.isfinite(self._ned_vz):
            return math.sqrt(self._ned_vx ** 2 + self._ned_vy ** 2 + self._ned_vz ** 2)
        return float("nan")

    def _set_position_target(self, x_enu: float, y_enu: float, z_enu: float, yaw: float = math.nan):
        x_ned, y_ned, z_ned = enu_pos_to_ned(x_enu, y_enu, z_enu)

        self._use_velocity_mode = False
        sp = TrajectorySetpoint()
        sp.timestamp = now_us(self)
        sp.position[:] = [x_ned, y_ned, z_ned]
        sp.velocity[:] = [math.nan, math.nan, math.nan]
        sp.acceleration[:] = [math.nan, math.nan, math.nan]
        if hasattr(sp, "jerk"):
            sp.jerk[:] = [math.nan, math.nan, math.nan]
        sp.yaw = yaw
        sp.yawspeed = 0.0
        self._last_sp = sp

    def hover(self,
              position_enu: Optional[tuple] = None,
              yaw: float = math.nan):
        """
        切换到位置模式，在给定 ENU 位置（x,y,z）悬停；若未指定则保持当前位置的水平位置并调整到 hover_alt 高度。
        """
        if position_enu is None:
            current = self._current_position_enu()
            if current is not None:
                x_enu, y_enu, _ = current
            else:
                x_enu = y_enu = 0.0
            z_enu = float(self.hover_alt)
        else:
            if len(position_enu) != 3:
                raise ValueError("position_enu 应为 (x, y, z) 三元组（ENU 坐标系）。")
            x_enu, y_enu, z_enu = map(float, position_enu)

        self._set_position_target(x_enu, y_enu, z_enu, yaw=yaw)

        target_alt = z_enu
        alt_tol = 0.05  # meters
        while True:
            current = self._current_position_enu()
            if current is not None:
                _, _, cur_z = current
                if math.isfinite(cur_z) and abs(cur_z - target_alt) <= alt_tol:
                    break
            self.get_logger().info(
                f"Hovering at ENU({x_enu:.2f}, {y_enu:.2f}, {z_enu:.2f})"
            )
            rclpy.spin_once(self, timeout_sec=0.05)

    def pos_cnt(self,
                position_enu: tuple,
                relative: bool = False,
                yaw: float = math.nan):
        """
        切换到位置模式，通过 ENU 位置指令控制。
        relative=True 时表示 position_enu 为相对位移（ENU），将基于当前位置计算目标点。
        """
        if len(position_enu) != 3:
            raise ValueError("position_enu 应为 (x, y, z) 三元组（ENU 坐标系）。")

        x_enu, y_enu, z_enu = map(float, position_enu)
        if relative:
            current = self._current_position_enu()
            if current is not None:
                cur_x, cur_y, cur_z = current
                x_enu += cur_x
                y_enu += cur_y
                z_enu += cur_z
            else:
                self.get_logger().warn("Relative position requested but current pose unknown; using absolute target.")

        self._set_position_target(x_enu, y_enu, z_enu, yaw=yaw)

        self.get_logger().info(
            f"Position control target ENU({x_enu:.2f}, {y_enu:.2f}, {z_enu:.2f})"
        )

    def forward(self, dist: float):
        """
        利用当前位置和 yaw 方向进行前进控制。
        """
        step = float(dist)
        if step == 0.0:
            return
        heading = self._heading_rad

        if math.isfinite(heading):
            # 将机体系前向位移按照 yaw 转换到 ENU 坐标
            delta_x = step * math.sin(heading)  # ENU x (East) 对应 NED y
            delta_y = step * math.cos(heading)  # ENU y (North) 对应 NED x
        else:
            delta_x = 0.0
            delta_y = 0.0
            self.get_logger().warn(
                "Forward requested before heading is available"
            )

        self.pos_cnt((delta_x, delta_y, 0.0), relative=True)

    def turn(self, rad: float):
        """
        原地转向（弧度），内部根据当前位置调用 pos_cnt 更新 yaw。
        """
        # 先将速度模式指令清零，结合当前位置速度反馈等待完全停止
        self.vel_cnt((0.0, 0.0, 0.0))
        vel_tol = 0.05  # m/s
        settle_deadline = self.get_clock().now() + Duration(seconds=1.0)
        settled = False
        while self.get_clock().now() < settle_deadline or settled:
            rclpy.spin_once(self, timeout_sec=0.05)
            current_speed = self._current_speed()
            if math.isfinite(current_speed) and current_speed <= vel_tol:
                settled = True
                break

        if not settled:
            self.get_logger().warn("Turn requested before vehicle fully stopped; proceeding after timeout.")

        yaw_delta = float(rad)
        if math.isfinite(self._heading_rad):
            target_yaw = self._heading_rad + yaw_delta
        else:
            target_yaw = yaw_delta
            self.get_logger().warn("Turn requested before heading is available; treating angle as absolute yaw.")

        current_pose = self._current_position_enu()
        if current_pose is None:
            current_pose = (0.0, 0.0, float(self.hover_alt))
            self.get_logger().warn("Turn requested before position is available; using hover altitude at origin.")

        self.pos_cnt(current_pose, relative=False, yaw=target_yaw)

    def vel_cnt(self,
                velocity_enu: tuple,
                yaw_rate_deg: float = 0.0):
        """
        切换到速度模式，通过 ENU 速度指令控制，无限持续直到外部更新。
        """
        if len(velocity_enu) != 3:
            raise ValueError("velocity_enu 应为 (vx, vy, vz) 三元组（ENU 坐标系）。")

        vx_enu, vy_enu, vz_enu = map(float, velocity_enu)
        yaw_rate = math.radians(float(yaw_rate_deg))

        self._use_velocity_mode = True
        sp = TrajectorySetpoint()
        sp.timestamp = now_us(self)
        sp.position[:] = [math.nan, math.nan, math.nan]
        vx_ned, vy_ned, vz_ned = enu_vel_to_ned(vx_enu, vy_enu, vz_enu)
        sp.velocity[:] = [vx_ned, vy_ned, vz_ned]
        sp.acceleration[:] = [math.nan, math.nan, math.nan]
        if hasattr(sp, "jerk"):
            sp.jerk[:] = [math.nan, math.nan, math.nan]
        sp.yaw = math.nan
        sp.yawspeed = yaw_rate
        self._last_sp = sp

    def reset_and_hover(self, hover_alt_m: Optional[float] = None):
        """
        兼容旧接口：先 take_off，再 hover 后切入速度模式悬停。
        """
        self.take_off(hover_alt_m=hover_alt_m)
        self.hover(position_enu=None)
        self.get_logger().info(f"Hover ready at ~{self.hover_alt:.2f} m (awaiting velocity commands).")
        # 切换至速度模式零速度保持
        self.vel_cnt((0.0, 0.0, 0.0))

    # ============================ High-level Env API ============================

    def step(self, velocity_enu: tuple, yaw_rate_deg: float = 0.0):
        """
        执行一步：直接调用 vel_cnt 发布 ENU 速度指令。
        """
        self.vel_cnt(velocity_enu, yaw_rate_deg=yaw_rate_deg)

    # ============================ Optional getters ============================

    def get_height_enu(self) -> float:
        """
        返回 ENU 下高度（米）。NED 的 z 向下为正 → ENU 高度 ≈ -z。
        """
        return float(-self.ned_z) if math.isfinite(self.ned_z) else float('nan')

    def latest_rgb(self):
        """
        若安装了 cv_bridge 且兼容，返回 numpy 数组，否则返回 ROS2 原始消息或 None
        """
        return self.rgb_img_np if self.rgb_img_np is not None else self.rgb_msg

    def latest_depth(self):
        """
        同上：返回 numpy 数组（H×W×1, float32）或 ROS2 消息或 None
        """
        return self.depth_img_np if self.depth_img_np is not None else self.depth_msg


# ============================ Main ============================

def main():
    rclpy.init()
    node = Px4IsaacSimVelEnv(
        node_name="px4_isaac_vel_env",
        hover_alt=1.0,
        keepalive_hz=20.0,
        subscribe_rgb=False,   # 需要时改 True（若 cv_bridge 不兼容将自动退化）
        subscribe_depth=False,
    )

    try:
        # 起飞并在 1m 高度悬停
        hover_alt_m=1.0
        node.take_off(hover_alt_m)
        node.hover()

        # settle_until = node.get_clock().now() + Duration(seconds=20.0)
        # while node.get_clock().now() < settle_until:
        #     rclpy.spin_once(node, timeout_sec=0.0)
        while node.ned_z >= -hover_alt_m:
            print("Up to setting height:{}")

        # 位置控制：向前（ENU x 正方向）移动 1 米
        node.get_logger().info("Moving forward 1 m using position control (pos_cnt).")
        node.pos_cnt((1.0, 0.0, 0.0), relative=True)

        node.get_logger().info("Moving forward with 1 m / s velocity control")
        node.vel_cnt((1.0, 0, 0))

        
        forward_until = node.get_clock().now() + Duration(seconds=5.0)
        while node.get_clock().now() < forward_until:
            rclpy.spin_once(node, timeout_sec=0.0)

        # 在新位置上悬停
        node.hover()
        node.get_logger().info("Position control demo completed; holding hover.")

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        # 可选：落地/解锁等收尾动作按需添加
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
