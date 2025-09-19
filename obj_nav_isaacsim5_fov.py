#!/usr/bin/env python

# For sim runing on the Isaacsim 5.0

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
import omni.usd
import math

from omni.isaac.core.world import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")                   # ROS 2 bridge
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.sensor import Camera
import omni.graph.core as og
import usdrt.Sdf

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
import numpy as np

# --- Global Constants ---
Env = ["Office", "Hospital"]
Obj = ["sugar_box", "thor_table"]
ROS_CAMERA_GRAPH_PATH = "/ROSCameraGraph"
CAMERA_PRIM_PATH = "/World/quadrotor/body/camera_fpv"
CAMERA_RESOLUTION = (640, 480)

class PegasusApp:
    def __init__(self, env_name="Hospital", obj_name="thor_table", flight_pos=np.array([0, 0, 1.0]), object_pos=np.array([0, 0, 2])):
        # Enable ROS2 bridge extension
        extensions.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        # Explicitly create the simulation world and assign it
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # Load environment
        print("="*50)
        self.pg.load_environment(SIMULATION_ENVIRONMENTS[env_name])
        print(f"load environment: {env_name}")
        print("="*50)

        # Configure and spawn the drone
        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0, "px4_autolaunch": True, "px4_dir": self.pg.px4_path, "px4_vehicle_model": "none_iris",
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        Multirotor(
            "/World/quadrotor", ROBOTS['Iris'], 0, flight_pos,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor
        )

        # Spawn an object in the world
        self._add_marker(obj_name, object_pos)

        # Add the FPV camera to the drone
        self._add_fpv_camera()

        # Build the ROS graph to publish camera data
        self._build_camera_graph()

        simulation_app.update()
        self.world.reset()
        self.stop_sim = False


    def _add_marker(self, obj_name, marker_pos):
        print("=" * 50)
        if obj_name == "thor_table":
            usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Props/Mounts/thor_table.usd"
            prim_utils.create_prim(
                prim_path="/World/thor_table", prim_type="Xform", usd_path=usd_url,
                position=marker_pos, orientation=np.array([0.7071, 0, 0, 0.7071]), scale=np.array([2.0, 2.0, 2.0])
            )
            print("Added Thor table.")
        elif obj_name == "sugar_box":
            usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd"
            prim_utils.create_prim(
                prim_path="/World/sugar_box", prim_type="Xform", usd_path=usd_url, position=marker_pos
            )
            print("Added sugar box.")
        print("=" * 50)

    def _add_fpv_camera(self):
        """
        Attaches a camera to the drone and sets its intrinsic properties for a specific FOV.
        """
        # ==================== MODIFIED CODE START ====================

        # Define camera intrinsic properties
        # Example: To achieve a 90-degree horizontal FOV, the focal length should be half the horizontal aperture.
        # FOV = 2 * atan(aperture / (2 * focal_length))
        
        # Let's set a desired horizontal FOV in degrees
        desired_fov_degrees = 90.0
        
        # We can fix the horizontal aperture (sensor width) to a common value, e.g., 24mm
        horizontal_aperture = 24.0
        
        # Calculate the required focal length
        focal_length = horizontal_aperture / (2 * math.tan(math.radians(desired_fov_degrees) / 2.0))
        
        # Calculate the vertical aperture based on the image aspect ratio
        vertical_aperture = horizontal_aperture * (CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION[0])

        print("-" * 50)
        print(f"Camera Settings for {desired_fov_degrees}-degree FOV:")
        print(f"  - Focal Length: {focal_length:.2f} mm")
        print(f"  - Horizontal Aperture: {horizontal_aperture:.2f} mm")
        print(f"  - Vertical Aperture: {vertical_aperture:.2f} mm")
        print("-" * 50)

        # Create the camera prim with all its properties (intrinsics) defined at creation.
        # This is a more robust way to set these values than using the high-level Camera class alone.
        camera_prim = prim_utils.create_prim(
            prim_path=CAMERA_PRIM_PATH,
            prim_type="Camera",
            attributes={
                "focalLength": focal_length,
                "horizontalAperture": horizontal_aperture,
                "verticalAperture": vertical_aperture,
                "clippingRange": (0.1, 1000.0), # Near and far clipping planes
            }
        )

        # Now, apply the high-level Isaac Sim Camera API to this prim for easy control
        self.camera = Camera(
            prim_path=CAMERA_PRIM_PATH,
            # The resolution of the output image is set here.
            # This is separate from the FOV.
            resolution=CAMERA_RESOLUTION
        )
        self.camera.initialize()
        
        # Position and orient the camera on the drone
        cam_pos = np.array([0.2, 0.0, 0.0])
        cam_orientation = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True)
        self.camera.set_local_pose(translation=cam_pos, orientation=cam_orientation)
        
        # ===================== MODIFIED CODE END =====================

    def _build_camera_graph(self):
        """
        Creates the OmniGraph for streaming the FPV camera's view via ROS2.
        This remains unchanged, as it correctly uses the camera prim defined above.
        """
        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": ROS_CAMERA_GRAPH_PATH, "evaluator_name": "push"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                    # 2. 设置分辨率（新增节点）
                    ("setViewportResolution", "isaacsim.core.nodes.IsaacSetViewportResolution"),
                    ("getViewportRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
                    ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                keys.CONNECT: [
                    # ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    # ("createViewport.outputs:execOut", "setViewportResolution.inputs:execIn"),
                    # ("createViewport.outputs:viewport", "setViewportResolution.inputs:viewport"),
                    # ("setViewportResolution.outputs:execOut", "getViewportRenderProduct.inputs:execIn"),
                    # ("setViewportResolution.outputs:viewport", "getViewportRenderProduct.inputs:viewport"),
                    # ("getViewportRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    # ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    # ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    # ("getViewportRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                    # ("getViewportRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                    # ("getViewportRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                    # 1. 执行流
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    ("createViewport.outputs:execOut", "setViewportResolution.inputs:execIn"),
                    ("setViewportResolution.outputs:execOut", "getViewportRenderProduct.inputs:execIn"),
                    ("getViewportRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    # 2. 渲染流
                    ("getViewportRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                    ("getViewportRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                    ("getViewportRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                ],
                keys.SET_VALUES: [
                    ("createViewport.inputs:name", "fpv_camera_viewport"),
                    ("setViewportResolution.inputs:viewport", "fpv_camera_viewport"),  # token
                    ("getViewportRenderProduct.inputs:viewport", "fpv_camera_viewport"),  # token

                    ("setViewportResolution.inputs:width", 640),
                    ("setViewportResolution.inputs:height", 480),
                    ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_PRIM_PATH)]),
                    ("cameraHelperRgb.inputs:frameId", "drone_fpv_camera"),
                    ("cameraHelperRgb.inputs:topicName", "rgb"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperDepth.inputs:frameId", "drone_fpv_camera"),
                    ("cameraHelperDepth.inputs:topicName", "depth"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                ],
            },
        )

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    import rclpy
    rclpy.init()
    pg_app = PegasusApp(
        env_name="Hospital",
        obj_name="thor_table",
        flight_pos=np.array([0, 0, 0.5]),
        object_pos=np.array([2, 0, 0])
    )
    pg_app.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
