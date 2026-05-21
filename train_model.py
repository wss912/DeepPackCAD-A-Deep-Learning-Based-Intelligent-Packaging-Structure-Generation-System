import json
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# 1. 基本配置
# =========================================================

DATA_PATH = "carton_dataset.csv"

MODEL_PATH = "carton_structure_model.pth"
SCALER_PATH = "carton_scaler.pkl"
META_PATH = "carton_model_meta.json"

REQUIRED_COLUMNS = [
    "product_L",
    "product_W",
    "product_H",
    "weight",
    "fragility",
    "category_id",
    "structure_id"
]


# =========================================================
# 2. 神经网络模型
# =========================================================

class CartonStructureNet(nn.Module):
    """
    包装结构分类模型

    输入：
        产品长、宽、高、重量、易碎等级、产品类别 one-hot

    输出：
        structure_id 分类结果
    """

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
# 3. 数据预处理
# =========================================================

def load_and_prepare_data(csv_path):
    """
    读取 CSV，并把 category_id 做 one-hot 编码。
    """

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"找不到数据文件：{csv_path}")

    df = pd.read_csv(csv_path)

    print("========== 原始数据预览 ==========")
    print(df.head())
    print()

    # 检查字段
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少字段：{col}")

    # 输入特征
    x_df = df[
        [
            "product_L",
            "product_W",
            "product_H",
            "weight",
            "fragility",
            "category_id"
        ]
    ].copy()

    # category_id 是类别变量，所以做 one-hot
    x_df = pd.get_dummies(
        x_df,
        columns=["category_id"],
        prefix="category"
    )

    # 标签
    y_series = df["structure_id"].astype(int)

    # 保存类别映射
    class_values = sorted(y_series.unique().tolist())
    class_to_index = {str(cls): idx for idx, cls in enumerate(class_values)}
    index_to_class = {str(idx): cls for idx, cls in enumerate(class_values)}

    y = y_series.map(lambda v: class_to_index[str(v)]).values

    feature_columns = x_df.columns.tolist()

    print("========== 模型输入特征 ==========")
    print(feature_columns)
    print()

    print("========== 结构类别 ==========")
    print("class_to_index:", class_to_index)
    print("index_to_class:", index_to_class)
    print()

    return x_df, y, feature_columns, class_to_index, index_to_class


# =========================================================
# 4. 训练函数
# =========================================================

def train():
    x_df, y, feature_columns, class_to_index, index_to_class = load_and_prepare_data(DATA_PATH)

    num_classes = len(class_to_index)
    input_dim = x_df.shape[1]

    if num_classes < 2:
        raise ValueError("structure_id 至少需要 2 个类别，否则无法训练分类模型。")

    # 标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    # 数据量很小时，stratify 可能报错，所以这里做保护
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    if len(y) >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if can_stratify else None
        )
    else:
        # 数据太少时，先全部用于训练
        x_train, x_test, y_train, y_test = x_scaled, x_scaled, y, y

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    model = CartonStructureNet(
        input_dim=input_dim,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 300

    print("========== 开始训练 ==========")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                test_logits = model(x_test)
                preds = torch.argmax(test_logits, dim=1)
                acc = (preds == y_test).float().mean().item()

            print(f"Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        train_preds = torch.argmax(model(x_train), dim=1)
        train_acc = (train_preds == y_train).float().mean().item()

        test_preds = torch.argmax(model(x_test), dim=1)
        test_acc = (test_preds == y_test).float().mean().item()

    print()
    print("========== 训练完成 ==========")
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 保存模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes
        },
        MODEL_PATH
    )

    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "feature_columns": feature_columns,
        "class_to_index": class_to_index,
        "index_to_class": index_to_class
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    print()
    print("========== 文件已保存 ==========")
    print(f"模型文件: {MODEL_PATH}")
    print(f"标准化文件: {SCALER_PATH}")
    print(f"模型信息文件: {META_PATH}")


if __name__ == "__main__":
    train()