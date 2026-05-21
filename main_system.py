from body_panels import generate_carton_dxf


def choose_carton_structure(category, fragility):
    """
    根据产品类别和易碎等级，选择包装结构。

    现在先用规则判断。
    后面你的深度学习模型，就是替代这个函数。
    """

    # 默认结构：普通插舌盒
    structure = {
        "top_tuck_type": "normal",
        "top_dust_type": "slant",

        "bottom_type": "tuck",
        "bottom_tuck_type": "normal",
        "bottom_dust_type": "slant"
    }

    # 如果产品需要更好的锁合，就用切缝锁定插舌 + 配套防尘翼
    if category in ["toothpaste", "cosmetic", "medicine"] or fragility >= 2:
        structure = {
            "top_tuck_type": "slit_lock",
            "top_dust_type": "slit_lock_dust",

            "bottom_type": "tuck",
            "bottom_tuck_type": "normal",
            "bottom_dust_type": "slit_lock_dust"
        }

    # 如果产品比较重，底部用 123 锁底
    if category in ["bottle", "heavy_product"] or fragility >= 3:
        structure = {
            "top_tuck_type": "slit_lock",
            "top_dust_type": "slit_lock_dust",

            "bottom_type": "123_lock",
            "bottom_tuck_type": "normal",
            "bottom_dust_type": "slit_lock_dust"
        }

    return structure


def run_system():
    """
    包装结构自动生成系统主入口。
    """

    # =====================================================
    # 1. 输入产品信息
    # 后面这里可以改成读取 STL 文件
    # =====================================================
    product_L = 120
    product_W = 40
    product_H = 180
    clearance = 3

    category = "toothpaste"
    weight = 0.3
    fragility = 1

    print("========== 产品输入信息 ==========")
    print(f"产品类别: {category}")
    print(f"产品尺寸: {product_L} × {product_W} × {product_H} mm")
    print(f"产品重量: {weight} kg")
    print(f"易碎等级: {fragility}")
    print()

    # =====================================================
    # 2. 根据产品信息选择包装结构
    # =====================================================
    structure = choose_carton_structure(category, fragility)

    print("========== 推荐包装结构 ==========")
    print(f"上插舌类型: {structure['top_tuck_type']}")
    print(f"上防尘翼类型: {structure['top_dust_type']}")
    print(f"底部结构: {structure['bottom_type']}")
    print(f"下插舌类型: {structure['bottom_tuck_type']}")
    print(f"下防尘翼类型: {structure['bottom_dust_type']}")
    print()

    # =====================================================
    # 3. 调用你的建模代码生成 DXF
    # =====================================================
    output_filename = f"output_{category}_{structure['bottom_type']}.dxf"

    generate_carton_dxf(
        filename=output_filename,

        product_L=product_L,
        product_W=product_W,
        product_H=product_H,
        clearance=clearance,

        top_tuck_type=structure["top_tuck_type"],
        top_dust_type=structure["top_dust_type"],

        bottom_type=structure["bottom_type"],
        bottom_tuck_type=structure["bottom_tuck_type"],
        bottom_dust_type=structure["bottom_dust_type"]
    )

    print()
    print("========== 系统运行完成 ==========")
    print(f"已生成包装展开图文件: {output_filename}")


if __name__ == "__main__":
    run_system()