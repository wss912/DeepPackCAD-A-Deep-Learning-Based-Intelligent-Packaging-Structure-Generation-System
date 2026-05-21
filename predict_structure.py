import json
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# 1. 文件路径
# =========================================================
# 使用当前文件所在目录作为基准路径，避免 Streamlit 从别的目录启动时找不到模型文件。

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "carton_structure_model.pth"
SCALER_PATH = BASE_DIR / "carton_scaler.pkl"
META_PATH = BASE_DIR / "carton_model_meta.json"


# =========================================================
# 2. structure_id 对应包装结构组合
# =========================================================
# 结构匹配原则：
# 普通插舌 normal        -> 普通斜边防尘翼 slant
# 切缝锁定插舌 slit_lock -> 切缝锁定配套防尘翼 slit_lock_dust
# 123 锁底 bottom_type=123_lock 时，底部不再实际绘制普通下插舌和下防尘翼。

STRUCTURE_MAP = {
    0: {
        "structure_name": "普通插舌折叠纸盒",
        "top_tuck_type": "normal",
        "top_dust_type": "slant",
        "bottom_type": "tuck",
        "bottom_tuck_type": "normal",
        "bottom_dust_type": "slant"
    },

    1: {
        "structure_name": "切缝锁定插舌折叠纸盒",
        # 上部：切缝锁定插舌 + 配套切缝锁定防尘翼
        "top_tuck_type": "slit_lock",
        "top_dust_type": "slit_lock_dust",

        # 底部：普通插舌底 + 普通斜边防尘翼
        # 这里不能写 slit_lock_dust，因为 bottom_tuck_type 是 normal。
        "bottom_type": "tuck",
        "bottom_tuck_type": "normal",
        "bottom_dust_type": "slant"
    },

    2: {
        "structure_name": "123锁底折叠纸盒",
        # 上部：切缝锁定插舌 + 配套切缝锁定防尘翼
        "top_tuck_type": "slit_lock",
        "top_dust_type": "slit_lock_dust",

        # 底部：123 锁底整体结构
        # 123 锁底不会实际使用 bottom_tuck_type 和 bottom_dust_type，
        # 这里保留 normal + slant 是为了界面显示不误导。
        "bottom_type": "123_lock",
        "bottom_tuck_type": "normal",
        "bottom_dust_type": "slant"
    }
}


# =========================================================
# 3. 模型结构
# 必须和 train_model.py 里的模型结构一致
# =========================================================

class CartonStructureNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 4. 结构参数自动校正
# =========================================================

def normalize_structure_config(structure):
    """
    对模型预测得到的结构配置做一次自动校正，避免插舌和防尘翼不配套。

    规则：
    1. 上插舌 normal -> top_dust_type = slant
    2. 上插舌 slit_lock -> top_dust_type = slit_lock_dust
    3. 底部 tuck + 下插舌 normal -> bottom_dust_type = slant
    4. 底部 tuck + 下插舌 slit_lock -> bottom_dust_type = slit_lock_dust
    5. 底部 123_lock -> bottom_tuck_type = normal, bottom_dust_type = slant
    """

    structure = structure.copy()

    # 上部自动匹配
    if structure.get("top_tuck_type") == "slit_lock":
        structure["top_dust_type"] = "slit_lock_dust"
    else:
        structure["top_dust_type"] = "slant"

    # 底部自动匹配
    if structure.get("bottom_type") == "tuck":
        if structure.get("bottom_tuck_type") == "slit_lock":
            structure["bottom_dust_type"] = "slit_lock_dust"
        else:
            structure["bottom_tuck_type"] = "normal"
            structure["bottom_dust_type"] = "slant"

    elif structure.get("bottom_type") == "123_lock":
        # 123 锁底不实际绘制普通下插舌和下防尘翼，界面显示为 normal + slant
        structure["bottom_tuck_type"] = "normal"
        structure["bottom_dust_type"] = "slant"

    return structure


# =========================================================
# 5. 加载模型
# =========================================================

def load_model():
    """
    加载训练好的模型、标准化器、特征列信息。
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"找不到标准化文件：{SCALER_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"找不到模型信息文件：{META_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    scaler = joblib.load(SCALER_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = CartonStructureNet(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, meta


# =========================================================
# 6. 构造模型输入
# =========================================================

def build_input_dataframe(
    product_L,
    product_W,
    product_H,
    weight,
    fragility,
    category_id,
    feature_columns
):
    """
    把单个产品参数转换成模型需要的输入格式。
    """

    df = pd.DataFrame(
        [
            {
                "product_L": product_L,
                "product_W": product_W,
                "product_H": product_H,
                "weight": weight,
                "fragility": fragility,
                "category_id": category_id
            }
        ]
    )

    # 和训练时一样，对 category_id 做 one-hot
    df = pd.get_dummies(
        df,
        columns=["category_id"],
        prefix="category"
    )

    # 对齐训练时保存的特征列
    # 比如训练时有 category_0, category_1, category_2
    # 预测时如果缺少某一列，就补 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df


# =========================================================
# 7. 预测函数
# =========================================================

def predict_structure(
    product_L,
    product_W,
    product_H,
    weight,
    fragility,
    category_id
):
    """
    输入产品参数，输出包装结构推荐结果。
    """

    model, scaler, meta = load_model()

    feature_columns = meta["feature_columns"]
    index_to_class = meta["index_to_class"]

    x_df = build_input_dataframe(
        product_L=product_L,
        product_W=product_W,
        product_H=product_H,
        weight=weight,
        fragility=fragility,
        category_id=category_id,
        feature_columns=feature_columns
    )

    x_scaled = scaler.transform(x_df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_index].item()

    # pred_index 是模型内部类别编号
    # structure_id 是你数据集里的真实结构编号
    structure_id = int(index_to_class[str(pred_index)])

    if structure_id not in STRUCTURE_MAP:
        raise ValueError(f"STRUCTURE_MAP 中没有 structure_id={structure_id} 的结构配置。")

    result = STRUCTURE_MAP[structure_id].copy()
    result = normalize_structure_config(result)

    result["structure_id"] = structure_id
    result["confidence"] = confidence

    return result


# =========================================================
# 8. 单独测试
# =========================================================

if __name__ == "__main__":
    result = predict_structure(
        product_L=120,
        product_W=40,
        product_H=180,
        weight=0.3,
        fragility=1,
        category_id=0
    )

    print("========== 预测结果 ==========")
    print(f"structure_id: {result['structure_id']}")
    print(f"结构名称: {result['structure_name']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"上插舌: {result['top_tuck_type']}")
    print(f"上防尘翼: {result['top_dust_type']}")
    print(f"底部结构: {result['bottom_type']}")
    print(f"下插舌: {result['bottom_tuck_type']}")
    print(f"下防尘翼: {result['bottom_dust_type']}")
