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
from pxr import Usd, UsdGeom, Sdf, Gf
import omni.usd
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.materials import OmniGlassMaterial

# ========= 新增：物理相关 =========
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
from omni.physx.scripts import utils as physx_utils
from pxr import PhysicsSchemaTools
# =================================

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
import numpy as np

# ---------- glass 探测与打印 ----------
from pxr import Usd, UsdGeom, UsdShade, Sdf

def wait_stage_fully_loaded():
    """等待 USD 引用/载入完成（很关键）。"""
    ctx = omni.usd.get_context()
    # 等待一切引用资源加载完成
    try:
        ctx.wait_for_loading()
    except Exception:
        pass

def print_children(root_path="/World"):
    """打印某个 prim 的直属子节点，方便确认真实路径。"""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root or not root.IsValid():
        print(f"[tree] invalid root: {root_path}")
        return
    print(f"[tree] children of {root_path}:")
    for c in root.GetChildren():
        print("   ", c.GetPath().pathString, "<", (c.GetTypeName() or "Prim"), ">")

# -------- robust glass detection & AABB --------

_GLASS_KEYS = ("glass", "window")  # 关键词，可按需再加

def _string_has_glass(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in _GLASS_KEYS)

def _get_bound_material(prim):
    """返回直接或collection绑定到 prim 的 UsdShade.Material（若有）。"""
    try:
        mb = UsdShade.MaterialBindingAPI(prim)
        # 优先 direct binding
        mat = mb.ComputeBoundMaterial()[0] if hasattr(mb, "ComputeBoundMaterial") else None
        if not mat or not mat:
            # 兼容：有些版本用 GetDirectBinding / GetMaterial
            db = mb.GetDirectBinding()
            mat = db.GetMaterial() if db else None
        return mat if (mat and mat.GetPrim() and mat.GetPrim().IsValid()) else None
    except Exception:
        return None

def _subtree_has_glass_material(root_prim):
    """自身或任意后代是否绑定材质名含 glass。"""
    stage = root_prim.GetStage()
    for p in stage.Traverse():
        if not p.GetPath().HasPrefix(root_prim.GetPath()):
            continue
        mat = _get_bound_material(p)
        if mat:
            # 只看材质 prim 的 name / path
            mp = mat.GetPrim().GetPath().pathString
            mn = mat.GetPrim().GetName()
            if _string_has_glass(mp) or _string_has_glass(mn):
                return True
    return False

def _compute_world_aabb(prim, time=Usd.TimeCode.Default()):
    """返回 (center, size)；若无几何可算则返回 None。"""
    bbox_cache = UsdGeom.BBoxCache(
        time,
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    try:
        aabb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()  # Gf.Range3d
        size = aabb.GetSize()
        if size[0] == 0 and size[1] == 0 and size[2] == 0:
            return None
        c = aabb.GetMidpoint()   # 注意：Range3d 用 GetMidpoint()
        return (Gf.Vec3d(c), Gf.Vec3d(size))
    except Exception:
        return None


def _first_mesh_nonzero_aabb(root_prim):
    """在 root_prim 子树里找第一个能算出非零 AABB 的 Mesh，返回 (meshPrim, (center,size)) 或 (None, None)。"""
    stage = root_prim.GetStage()
    for p in stage.Traverse():
        if not p.GetPath().HasPrefix(root_prim.GetPath()):
            continue
        if p.IsA(UsdGeom.Mesh):
            res = _compute_world_aabb(p)
            if res is not None:
                return p, res
    return None, None

def list_glass_like_robust(root="/World/layout"):
    """
    规则：
    1) prim 名称/路径含 glass/window，或
    2) 自身/任意后代绑定的材质名含 glass
    则认为“玻璃相关”，并输出世界 AABB（优先整棵子树，否则找一个能算出AABB的 Mesh）。
    """
    stage = omni.usd.get_context().get_stage()
    r = stage.GetPrimAtPath(root)
    if not r or not r.IsValid():
        print(f"[glass] invalid root: {root}")
        return []

    out = []
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(r.GetPath()):
            continue

        # 快速筛：名字/路径命中 或 材质命中
        name_or_path_hit = _string_has_glass(prim.GetName()) or _string_has_glass(prim.GetPath().pathString)
        mat_hit = _subtree_has_glass_material(prim) if not name_or_path_hit else True
        if not (name_or_path_hit or mat_hit):
            continue

        entry = {
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
            "center": None,
            "size": None,
            "source": None,   # 'subtree' or 'mesh'
        }

        # 先试整棵子树
        aabb = _compute_world_aabb(prim)
        if aabb is not None:
            c, s = aabb
            entry["center"] = (float(c[0]), float(c[1]), float(c[2]))
            entry["size"]   = (float(s[0]), float(s[1]), float(s[2]))
            entry["source"] = "subtree"
        else:
            # 找第一个能算 AABB 的 Mesh
            m, aabb = _first_mesh_nonzero_aabb(prim)
            if m and aabb:
                c, s = aabb
                entry["center"] = (float(c[0]), float(c[1]), float(c[2]))
                entry["size"]   = (float(s[0]), float(s[1]), float(s[2]))
                entry["source"] = f"mesh:{m.GetPath().pathString}"

        out.append(entry)
    return out

def print_glass_like_robust(root="/World/layout"):
    wait_stage_fully_loaded()
    items = list_glass_like_robust(root)
    if not items:
        print(f"[glass] none under {root}")
        return
    print(f"[glass] {len(items)} glass-like prim(s) under {root}:")
    for it in items:
        line = f"  {it['path']} < {it['type']} >"
        if it["center"] is not None:
            line += f"\n     center: {it['center']}, size(LWH): {it['size']} [{it['source']}]"
        print(line)


# ========= depth-only occluder helpers =========

def _ensure_xform(stage, path: str):
    """确保 path 上存在一个 Xform prim（若无则 Define）。"""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(path))  # 正确的 Define 用法
        prim = stage.GetPrimAtPath(path)
    return prim

def build_glass_cubes_only(items,
                           root="/World/_visual_glass",
                           mtl_path="/World/Looks/_AutoGlass"):
    """
    在筛出的玻璃类物体位置创建“透明玻璃”的 Cube，并用 OmniGlass 材质绑定。
    需求来源：Isaac Sim 5.0 Objects.Cube & OmniGlassMaterial API。
    """
    stage = omni.usd.get_context().get_stage()

    # 1) 确保根 Xform 存在（修正你之前错误的 Usd.Prim.Define）
    _ensure_xform(stage, root)

    # 2) 构造批量创建所需的 arrays
    paths, sizes, positions, scales = [], [], [], []
    for i, it in enumerate(items):
        center = it["center"]
        size = it["size"]
        # Cube path
        p = f"{root}/glass_{i:03d}"
        paths.append(p)
        # Cube 的“几何边长”（标量），各向异性尺寸用 scales 给
        sizes.append([1.0])  # 每个 cube 的 base edge 长度设为 1，再用 scales 拉伸
        positions.append([float(center[0]), float(center[1]), float(center[2])])
        scales.append([float(size[0]), float(size[1]) * 0.01, float(size[2])])

    if not paths:
        print("[glass] 没有可生成的玻璃 cube。")
        return

    # 3) 批量创建/覆盖 Cube：
    cubes = Cube(
        paths=paths,
        sizes=sizes,
        positions=positions,
        scales=scales,
        reset_xform_op_properties=True,   # 解决 xformOp 断言问题
    )
    # 以上用法与文档一致（包括 reset_xform_op_properties / existing_prim_behavior）:contentReference[oaicite:3]{index=3}

    # 4) 创建/获取 OmniGlass 材质（5.0 文档：用构造函数传一个 prim 路径字符串/列表）
    glass_mtl = OmniGlassMaterial(paths=mtl_path)
    #    ↓↓↓ 按文档给参数名：glass_color / reflection_color / glass_ior / thin_walled / enable_opacity / depth / frosting_roughness
    glass_mtl.set_input_values(name="glass_color",       values=[1.0, 1.0, 1.0])
    glass_mtl.set_input_values(name="reflection_color",  values=[1.0, 1.0, 1.0])
    glass_mtl.set_input_values(name="glass_ior",         values=[1.491])
    glass_mtl.set_input_values(name="thin_walled",       values=[False])
    glass_mtl.set_input_values(name="enable_opacity",    values=[False])   # 与 UI 默认一致
    # 下面两个按需调；给个合理初值
    glass_mtl.set_input_values(name="depth",             values=[0.001])
    glass_mtl.set_input_values(name="frosting_roughness",values=[0.0])
    # 参数名与调用范式符合 5.0 文档（OmniGlassMaterial & set_input_values）。:contentReference[oaicite:4]{index=4}

    # 5) 绑定材质（Objects API 的 apply_visual_materials 示例就是这么干的）
    cubes.apply_visual_materials(glass_mtl)  # 支持单个或同长度数组
    # 绑定方法及示例参见 Objects 文档。:contentReference[oaicite:5]{index=5}

    print(f"[glass] 已创建 {len(paths)} 个 OmniGlass cube 并完成材质绑定。")







# ---------- end ----------


# --- Global Constants ---
Env = ["Office", "Hospital"]
Obj = ["sugar_box", "thor_table", "manzana_apple"]
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
        # self.pg.load_environment(usd_path="/home/shaw/devspace/obj_nav_sim/XR_Content_NVD@10010/Assets/XR/Stages/Indoor/Modern_House_XR_modified.usd")
        # self.pg.load_environment(usd_path='/home/shaw/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/Indoor/Warehouse_XR.usd')
        print(f"load environment: {env_name}")
        print("="*50)

        # ======= 新增：确保物理场景与地面存在 =======
        self._ensure_physics()
        # ==========================================

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

        print("="*50)
        print_glass_like_robust("/World/layout")
        print("="*50)

        # items = list_glass_like_robust("/World/layout")
        # build_glass_cubes_only(items)  # 透明度可调：0.05~0.2
        from build_glass_cube import build_glass_cubes_from_xforms
        build_glass_cubes_from_xforms("/World/layout", "/World/_visual_glass", thickness=0.02)

        simulation_app.update()  # 刷一下



    def _add_marker(self, obj_name, marker_pos):
        print("=" * 50)
        spawned_prim_path = None
        if obj_name == "thor_table":
            usd_url = "/home/shaw/devspace/SimulatorSetup/submodules/PegasusSimulator/examples/obj_nav_sim/assets/thor_table.usd"
            prim_utils.create_prim(
                prim_path="/World/thor_table", prim_type="Xform", usd_path=usd_url,
                position=marker_pos, orientation=np.array([0.7071, 0, 0, 0.7071]), scale=np.array([1.0, 1.0, 1.0])
            )
            print("Added Thor table.")
            spawned_prim_path = "/World/thor_table"
        elif obj_name == "sugar_box":
            usd_url = "/home/shaw/devspace/SimulatorSetup/submodules/PegasusSimulator/examples/obj_nav_sim/assets/sugar_box.usd"
            prim_utils.create_prim(
                prim_path="/World/sugar_box", prim_type="Xform", usd_path=usd_url, position=marker_pos
            )
            print("Added sugar box.")
            spawned_prim_path = "/World/sugar_box"
        elif obj_name == "manzana_apple":
            usd_url = "/home/shaw/devspace/SimulatorSetup/submodules/PegasusSimulator/examples/obj_nav_sim/assets/Manzana.usd"
            prim_utils.create_prim(
                prim_path="/World/manzana_apple", prim_type="Xform", usd_path=usd_url, 
                position=marker_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]), # W, X, Y, Z
                scale=np.array([0.01, 0.01, 0.01]),
            )
            print("Added manzana_apple.")
            spawned_prim_path = "/World/manzana_apple"
        print("=" * 50)

        # ======= 新增：为导入物体启用物理（刚体+碰撞） =======
        if spawned_prim_path:
            # 若想作为静态障碍物：dynamic=False（不受重力，仅碰撞）
            self._make_rigid_with_collider(spawned_prim_path, dynamic=True, approximation="convexDecomposition")
        # ====================================================


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

    # ====================== 新增：物理辅助函数 ======================
    def _ensure_physics(self):
        """
        确保场景中存在 Physics Scene（含重力设置）与地面。
        - 重力方向：-Z
        - 重力大小：981 cm/s^2（Isaac 默认单位为厘米）
        - 物理引擎：开启 CCD 和 TGS 求解器以提高稳定性
        - 地面：/World/ground
        """
        stage = omni.usd.get_context().get_stage()
        # DefaultPrim
        if not stage.GetDefaultPrim():
            UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
            stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        # Physics Scene
        physics_prim_path = Sdf.Path("/World/physicsScene")
        if not stage.GetPrimAtPath(physics_prim_path):
            scene = UsdPhysics.Scene.Define(stage, physics_prim_path)
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(981.0)  # cm/s^2
            # PhysX 选项
            PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(str(physics_prim_path)))
            physx = PhysxSchema.PhysxSceneAPI.Get(stage, str(physics_prim_path))
            physx.CreateEnableCCDAttr(True)
            physx.CreateEnableStabilizationAttr(True)
            physx.CreateSolverTypeAttr("TGS")
        else:
            scene = UsdPhysics.Scene.Get(stage, physics_prim_path)
            if scene:
                scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
                scene.CreateGravityMagnitudeAttr().Set(981.0)

        # Ground（若环境已带地面，可跳过；否则创建）
        if not stage.GetPrimAtPath("/World/ground"):
            PhysicsSchemaTools.addGroundPlane(
                stage, "/World/ground", "Z", 1000, Gf.Vec3f(0, 0, 0), Gf.Vec3f(1)
            )

    def _make_rigid_with_collider(self, root_prim_path: str, dynamic: bool = True, approximation: str = "convexDecomposition"):
        """
        给指定 prim（通常是一个 Xform，内部包含 Mesh）启用刚体和碰撞体。
        - dynamic=True  -> 动态刚体（受重力），False -> 静态（仅作为场景障碍）
        - approximation 可选：'convexDecomposition' | 'convexHull' | 'none'
          * 动态刚体建议使用 'convexDecomposition' 或 'convexHull'
          * 'none' 使用三角网格，通常只适合静态
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(root_prim_path)
        if not prim or not prim.IsValid():
            carb.log_error(f"[physics] prim not found: {root_prim_path}")
            return

        # 1) 根节点刚体（动态或静态）
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False)
        else:
            # 静态：不加 RigidBody，只加 Collider
            pass

        # 2) 为子网格添加 collider
        approx = approximation
        if not dynamic and approximation.lower() == "convexdecomposition":
            approx = "convexHull"

        for sub in stage.Traverse():
            p = sub.GetPath().pathString
            if not p.startswith(root_prim_path):
                continue
            if sub.IsA(UsdGeom.Mesh):
                try:
                    if approx.lower() == "none":
                        # 使用三角网格（仅建议静态）
                        physx_utils.setCollider(sub, approximationShape="none", kinematic=False)
                    else:
                        physx_utils.setCollider(sub, approximationShape=approx, kinematic=False)
                except Exception as e:
                    carb.log_warn(f"[physics] setCollider failed on {p}: {e}")
    # ====================== 新增函数结束 ======================


def main():
    import rclpy
    rclpy.init()
    pg_app = PegasusApp(
        env_name="Office",
        obj_name="manzana_apple",
        flight_pos=np.array([-22, 9.3, 0.5]),
        object_pos=np.array([2, 0, 0.5])
    )
    pg_app.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
