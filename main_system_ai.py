from predict_structure import predict_structure
from body_panels import generate_carton_dxf


def run_ai_system():
    """
    深度学习包装结构生成系统主入口。
    """

    # =====================================================
    # 1. 输入产品信息
    # 后面这里可以换成 STL 自动读取尺寸
    # =====================================================

    product_L = 120
    product_W = 40
    product_H = 180
    clearance = 3

    weight = 0.3
    fragility = 1
    category_id = 0

    print("========== 产品输入信息 ==========")
    print(f"产品尺寸: {product_L} × {product_W} × {product_H} mm")
    print(f"产品重量: {weight} kg")
    print(f"易碎等级: {fragility}")
    print(f"产品类别编号: {category_id}")
    print()

    # =====================================================
    # 2. 深度学习模型预测包装结构
    # =====================================================

    structure = predict_structure(
        product_L=product_L,
        product_W=product_W,
        product_H=product_H,
        weight=weight,
        fragility=fragility,
        category_id=category_id
    )

    print("========== 深度学习预测结果 ==========")
    print(f"预测结构编号: {structure['structure_id']}")
    print(f"结构名称: {structure['structure_name']}")
    print(f"预测置信度: {structure['confidence']:.4f}")
    print(f"上插舌类型: {structure['top_tuck_type']}")
    print(f"上防尘翼类型: {structure['top_dust_type']}")
    print(f"底部结构: {structure['bottom_type']}")
    print(f"下插舌类型: {structure['bottom_tuck_type']}")
    print(f"下防尘翼类型: {structure['bottom_dust_type']}")
    print()

    # =====================================================
    # 3. 调用参数化 CAD 模块生成 DXF
    # =====================================================

    output_filename = f"ai_output_structure_{structure['structure_id']}.dxf"

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
    print(f"已生成 DXF 文件: {output_filename}")


if __name__ == "__main__":
    run_ai_system()