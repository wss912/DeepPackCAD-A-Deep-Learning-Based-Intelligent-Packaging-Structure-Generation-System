#stl参数读取模块
import trimesh
import os
def get_stl_dimensions(file_path):
    mesh = trimesh.load(file_path)
    bbox=mesh.bounding_box.extents
    l, w, h= sorted(bbox, reverse=True)
    print("产品尺寸：")
    print(f"长 L = {l:.2f} mm")
    print(f"宽 W = {w:.2f} mm")
    print(f"高 H = {h:.2f} mm")
def create_stl():
    os.makedirs("stl", exist_ok=True)
    toothpaste = trimesh.creation.box(extents=[180, 45, 35])
    toothpaste.export("stl/toothpaste.stl")
    print("牙膏测试模型已生成：stl/toothpaste.stl")
create_stl()
get_stl_dimensions("stl/toothpaste.stl")
