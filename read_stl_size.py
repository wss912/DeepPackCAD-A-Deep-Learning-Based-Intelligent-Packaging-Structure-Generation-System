import trimesh


def read_stl_size(stl_path):
    """
    读取 STL 文件的包围盒尺寸。

    返回：
        product_L, product_W, product_H
    """

    mesh = trimesh.load(stl_path, force="mesh")

    # 如果 STL 被读成 Scene，就合并里面的 mesh
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geometries)

    # extents = [x方向长度, y方向宽度, z方向高度]
    x_size, y_size, z_size = mesh.extents

    product_L = float(x_size)
    product_W = float(y_size)
    product_H = float(z_size)

    return product_L, product_W, product_H