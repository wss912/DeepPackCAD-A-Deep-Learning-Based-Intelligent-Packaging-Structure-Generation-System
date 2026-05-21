import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================
# 1. 读取数据
# =========================
data = pd.read_csv("carton_dataset.csv")

feature_cols = [
    "product_L",
    "product_W",
    "product_H",
    "weight",
    "fragility",
    "category_id"
]

label_col = "structure_id"

X = data[feature_cols].values
y = data[label_col].values


# =========================
# 2. 划分训练集、验证集、测试集
# =========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=1/9,
    random_state=42,
    stratify=y_trainval
)


# =========================
# 3. 数据标准化
# 只对前4个连续特征标准化：
# product_L, product_W, product_H, weight
# =========================
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, 0:4] = scaler.fit_transform(X_train[:, 0:4])
X_val_scaled[:, 0:4] = scaler.transform(X_val[:, 0:4])
X_test_scaled[:, 0:4] = scaler.transform(X_test[:, 0:4])


# =========================
# 4. 转成 PyTorch 张量
# =========================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# =========================
# 5. 定义神经网络模型
# =========================
class CartonMLP(nn.Module):
    def __init__(self, input_dim=6, num_classes=3):
        super(CartonMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


num_classes = len(np.unique(y))
model = CartonMLP(input_dim=6, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================
# 6. 计算准确率函数
# =========================
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


# =========================
# 7. 模型训练
# 这里就是曲线数据的来源
# =========================
epochs = 30

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

for epoch in range(1, epochs + 1):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, val_loader, criterion)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"train_acc={train_acc:.4f} | "
        f"val_acc={val_acc:.4f}"
    )


# =========================
# 8. 保存训练过程数据
# 这个 CSV 就是你论文曲线的数据来源
# =========================
history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False, encoding="utf-8-sig")


# =========================
# 9. 绘制 loss 曲线
# =========================
plt.figure()
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 10. 绘制 accuracy 曲线
# =========================
plt.figure()
plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 11. 测试集评价
# =========================
test_loss, test_acc = evaluate(model, test_loader, criterion)

print("\n测试集结果：")
print("test_loss =", test_loss)
print("test_acc =", test_acc)


# =========================
# 12. 混淆矩阵
# =========================
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

cm = confusion_matrix(all_labels, all_preds)

print("\n混淆矩阵：")
print(cm)

print("\n分类报告：")
target_names = [
    "Normal Tuck",
    "Slit-lock Tuck",
    "1-2-3 Lock-bottom"
]

print(classification_report(all_labels, all_preds, target_names=target_names))


# =========================
# 13. 画混淆矩阵
# =========================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=30, ha="right")
plt.yticks(tick_marks, target_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 14. 保存模型
# =========================
torch.save(model.state_dict(), "carton_structure_model.pth")

print("\n训练完成，已生成：")
print("training_history.csv")
print("loss_curve.png")
print("accuracy_curve.png")
print("confusion_matrix.png")
print("carton_structure_model.pth")