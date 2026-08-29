import os
import sys

# ------------------------------------------------------------------
# 把项目根目录（low_data/ 的上一级）加入 sys.path
#
# 这样无论：
# python low_data/main.py
#
# 还是：
# cd low_data
# python main.py
#
# 都可以正常导入：
# common/
# resnet18/
#
# macOS / Windows 通用
# ------------------------------------------------------------------

_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_FILE_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from common.data import get_dataloaders
from common.engine import train_one_epoch, evaluate, predict

# 直接复用原来的 ResNet18
from resnet18.resnet18 import ResNet18


# ============================================================
# 固定随机种子
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 自动选择设备
# ============================================================

def get_device():

    # Windows / Linux + NVIDIA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Mac Apple Silicon
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    # 没有 GPU
    else:
        return torch.device("cpu")


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 0. Experiment Config
    # ========================================================

    seed = 42

    # 本次 low-data 的核心设置
    samples_per_class = 100

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 32

    set_seed(seed)

    print("=" * 60)
    print("Low-data ResNet18 Experiment")
    print("=" * 60)

    print(f"Seed: {seed}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")

    print()


    # ========================================================
    # 1. Data
    # ========================================================

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        samples_per_class=samples_per_class,
        seed=seed
    )

    print()
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print()


    # ========================================================
    # 2. Device
    # ========================================================

    device = get_device()

    print("使用设备:", device)

    print()


    # ========================================================
    # 3. Model
    # ========================================================

    model = ResNet18(
        num_classes=11
    ).to(device)


    # ========================================================
    # 4. Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # 5. Optimizer
    # ========================================================

    # 和 Full-data ResNet18 baseline 保持一致
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # ========================================================
    # 6. Training Config
    # ========================================================

    best_val_accuracy = 0.0
    best_epoch = 0

    # 注意：
    # 不要覆盖 resnet18/best_resnet18.pth
    # 基于当前文件位置确定 low_data/ 目录，避免 cwd 变化时写到嵌套子目录
    save_path = os.path.join(
        _CURRENT_FILE_DIR,
        f"best_resnet18_{samples_per_class}perclass.pth"
    )

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


    # ========================================================
    # 7. Start Timing
    # ========================================================

    start_time = time.time()


    # ========================================================
    # 8. Training
    # ========================================================

    for epoch in range(num_epochs):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )


        # ----------------------------------------------------
        # 保存历史数据
        # ----------------------------------------------------

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )


        # ----------------------------------------------------
        # 保存最佳模型
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"保存当前最佳模型："
                f"Epoch {best_epoch}, "
                f"Val Acc: {best_val_accuracy:.4f}"
            )


        # ----------------------------------------------------
        # 当前 Epoch 输出
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
        )


    # ========================================================
    # 9. Load Best Model
    # ========================================================

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True
        )
    )

    print(
        f"\n已加载最佳模型："
        f"Epoch {best_epoch}, "
        f"Val Acc: {best_val_accuracy:.4f}"
    )


    # ========================================================
    # 10. Test
    # ========================================================

    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}"
    )


    # ========================================================
    # 11. Predict
    # ========================================================

    true_labels, predictions = predict(
        model=model,
        data_loader=test_loader,
        device=device
    )


    # ========================================================
    # 12. Macro-F1
    # ========================================================

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    print(
        f"Test Macro-F1: {macro_f1:.4f}"
    )


    # ========================================================
    # 13. Classification Report
    # ========================================================

    class_names = [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-left",
        "lung-right",
        "pancreas",
        "spleen"
    ]

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            digits=4
        )
    )


    # ========================================================
    # 14. Runtime
    # ========================================================

    elapsed_time = time.time() - start_time

    print(
        f"总运行时间: "
        f"{elapsed_time / 60:.2f} min"
    )


    # ========================================================
    # 15. Loss Curve
    # ========================================================

    epochs = range(
        1,
        num_epochs + 1
    )

    plt.figure()

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss",
        marker="o"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"ResNet18 Low-data "
        f"({samples_per_class}/class) "
        f"Loss"
    )

    plt.legend()
    plt.grid()

    plt.show()


    # ========================================================
    # 16. Accuracy Curve
    # ========================================================

    plt.figure()

    plt.plot(
        epochs,
        train_accuracies,
        label="Train Accuracy",
        marker="o"
    )

    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"ResNet18 Low-data "
        f"({samples_per_class}/class) "
        f"Accuracy"
    )

    plt.legend()
    plt.grid()

    plt.show()


    # ========================================================
    # 17. Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        f"ResNet18 Low-data "
        f"({samples_per_class}/class) "
        f"Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Program Entry
# ============================================================

if __name__ == "__main__":
    main()