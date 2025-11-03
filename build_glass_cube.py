from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import omni.usd
from isaacsim.core.experimental.materials import OmniGlassMaterial

def _sanitize_usd_name(name: str) -> str:
    """把任意字符串变成 USD 安全的标识：只保留字母数字._-，其它转 '_'；不能以数字开头。"""
    s = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (name or ""))
    if not s:
        s = "unnamed"
    if s[0].isdigit():
        s = "_" + s
    return s

def _get_or_create_glass_material(mtl_path="/World/Looks/_AutoGlass"):
    stage = omni.usd.get_context().get_stage()
    _mtl_obj = OmniGlassMaterial(paths=mtl_path)  # 若存在则包裹，若不存在则创建
    _mtl_obj.set_input_values(name="glass_color",        values=[1.0, 0.0, 1.0])
    _mtl_obj.set_input_values(name="reflection_color",   values=[1.0, 1.0, 1.0])
    _mtl_obj.set_input_values(name="glass_ior",          values=[1.491])
    _mtl_obj.set_input_values(name="thin_walled",        values=[False])
    _mtl_obj.set_input_values(name="enable_opacity",     values=[False])
    _mtl_obj.set_input_values(name="depth",              values=[0.001])
    _mtl_obj.set_input_values(name="frosting_roughness", values=[0.0])

    usd_mtl = UsdShade.Material.Get(stage, Sdf.Path(mtl_path))
    if not usd_mtl:
        usd_mtl = UsdShade.Material(stage.GetPrimAtPath(mtl_path))
    return usd_mtl, _mtl_obj


def build_glass_cubes_from_xforms(root="/World/layout",
                                  out_root="/World/_visual_glass",
                                  thickness=None):
    """为 root 子树中名字含 glass/window 的 prim，各生成一个与 OBB 对齐的 Cube，命名为 glass_<原名>[可带编号]。"""
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()

    root_prim = stage.GetPrimAtPath(root)
    if not root_prim or not root_prim.IsValid():
        print(f"[glass-cubes] invalid root: {root}")
        return

    # 确保输出根
    if not stage.GetPrimAtPath(out_root):
        UsdGeom.Xform.Define(stage, out_root)

    # BBoxCache：拿“有朝向的包围盒” (OBB 的 matrix + range)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )

    # 材质（一次性准备）
    usd_mtl, _ = _get_or_create_glass_material("/World/Looks/_AutoGlass")

    # 为避免重复名，做个计数器
    name_counter = {}

    rng = Usd.PrimRange(root_prim)
    built = 0
    for prim in rng:
        
        name = prim.GetName().lower()
        path_s = prim.GetPath().pathString.lower()
        if ("glass" not in name and "window" not in name
                and "glass" not in path_s and "window" not in path_s):
            continue

        bbox = cache.ComputeWorldBound(prim)
        box  = bbox.ComputeAlignedBox()    
        
        if box.IsEmpty():
            continue

        c_local = box.GetMidpoint()
        size    = box.GetSize()               # (L, W, H)

        # === 关键：名字改成 glass_<原始名字>[可带编号] ===
        base_name = _sanitize_usd_name(prim.GetName())
        target = f"glass_{base_name}"
        if target in name_counter:
            name_counter[target] += 1
            target = f"{target}_{name_counter[target]}"
        else:
            name_counter[target] = 1

        cube_xf_path = f"{out_root}/{target}"
        xform = UsdGeom.Xform.Define(stage, cube_xf_path)
        T_local = Gf.Matrix4d().SetTranslate(c_local)
        xf_op   = xform.AddXformOp(UsdGeom.XformOp.TypeTransform)
        xf_op.Set(T_local)

        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        if thickness is not None:
            arr = [sx, sy, sz]
            k = min(range(3), key=lambda i: arr[i])  # 最薄轴夹到 thickness
            arr[k] = float(thickness)
            sx, sy, sz = arr
        xf_scale = xform.AddXformOp(UsdGeom.XformOp.TypeScale)
        xf_scale.Set(Gf.Vec3d(sx, sy, sz))

        # 创建单位立方体并缩放
        cube = UsdGeom.Cube.Define(stage, f"{cube_xf_path}/cube")
        cube.CreateSizeAttr(1.0)
        # 绑定 OmniGlass 材质
        cube_prim = cube.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(usd_mtl)

        built += 1

    print(f"[glass-cubes] built {built} cubes under {out_root}") 