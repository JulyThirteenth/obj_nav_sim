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
        take_off(): 切入 Offboard 并解锁
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

        # “最近一次速度 setpoint”，用于定时器保活重复发布
        self._last_sp: Optional[TrajectorySetpoint] = None
        self._use_velocity_mode = False  # 起飞阶段先用 position 模式

        # ---------- Timer 保活 ----------
        self._timer = self.create_timer(self.keepalive_period, self._on_timer)

    # ============================ Callbacks ============================

    def _on_lpos(self, msg: VehicleLocalPosition):
        self.ned_x = float(msg.x)
        self.ned_y = float(msg.y)
        self.ned_z = float(msg.z)  # NED：向下为正

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
        
        
    def take_off(self):
        """
        进入 Offboard 并解锁。
        """
        self._use_velocity_mode = False  # 起飞阶段用 position
        self._last_sp = None
        while not self.armed or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.enter_offboard_and_arm()
            self.get_logger().info("Waiting for armed + offboard...")

        self.get_logger().info("Offboard + arm complete.")

    def hover(self,
              position_enu: Optional[tuple] = None,
              yaw: float = math.nan):
        """
        切换到位置模式，在给定 ENU 位置（x,y,z）悬停；若未指定则保持当前位置的水平位置并调整到 hover_alt 高度。
        """
        if position_enu is None:
            if math.isfinite(self.ned_x) and math.isfinite(self.ned_y):
                # NED -> ENU
                x_enu = float(self.ned_y)
                y_enu = float(self.ned_x)
            else:
                x_enu = y_enu = 0.0
            z_enu = float(self.hover_alt)
        else:
            if len(position_enu) != 3:
                raise ValueError("position_enu 应为 (x, y, z) 三元组（ENU 坐标系）。")
            x_enu, y_enu, z_enu = map(float, position_enu)

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

        self.get_logger().info(
            f"Hovering at ENU({x_enu:.2f}, {y_enu:.2f}, {z_enu:.2f})"
        )

    def reset_and_hover(self):
        """
        兼容旧接口：先 take_off，再 hover 后切入速度模式悬停。
        """
        self.take_off()
        self.hover(position_enu=None)
        self.get_logger().info(f"Hover ready at ~{self.hover_alt:.2f} m (awaiting velocity commands).")
        # 切换至速度模式零速度保持
        self.vel_cnt((0.0, 0.0, 0.0))

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
        node.take_off()
        node.hover()

        # # 示例：每 1s 向前飞一下、再原地，循环 5 次
        # for i in range(5):
        #     node.get_logger().info(f"[DEMO] Forward pulse {i+1}/5")
        #     node.step((0.3, 0.0, 0.0))
        #     t_end = node.get_clock().now() + Duration(seconds=1.0)
        #     while node.get_clock().now() < t_end:
        #         rclpy.spin_once(node, timeout_sec=0.0)

        #     node.step((0.0, 0.0, 0.0))  # 停止：清零速度指令
        #     t_end = node.get_clock().now() + Duration(seconds=1.0)
        #     while node.get_clock().now() < t_end:
        #         rclpy.spin_once(node, timeout_sec=0.0)

        # node.get_logger().info("Demo finished. Node keeps Offboard hover until process exit.")
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        # 可选：落地/解锁等收尾动作按需添加
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
