import os
import sys

# ------------------------------------------------------------------
# 把项目根目录（resnet18/ 的上一级）加入 sys.path，
# 保证无论以「python resnet18/main.py」还是
# 「cd resnet18 && python main.py」运行，都能导入顶层包 common/
# 和通过包名形式导入 resnet18.* 自身模块（跨平台，macOS/Windows 一致）
# ------------------------------------------------------------------
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_FILE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import time


from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from common.data import get_dataloaders
from common.engine import train_one_epoch, evaluate, predict

from resnet18.resnet18 import ResNet18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():

    set_seed(42)

    # =========================
    # 1. Data
    # =========================

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=64
    )


    # =========================
    # 2. Device
    # =========================

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print("使用设备:", device)


    # =========================
    # 3. Model
    # =========================

    model = ResNet18(
        num_classes=11
    ).to(device)


    # =========================
    # 4. Loss
    # =========================

    criterion = nn.CrossEntropyLoss()


    # =========================
    # 5. Optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # =========================
    # 6. Config
    # =========================

    num_epochs = 10

    best_val_accuracy = 0.0
    best_epoch = 0

    save_path = "resnet18/best_resnet18.pth"

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


    # =========================
    # 7. Training
    # =========================

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)


        # =========================
        # 保存最佳模型
        # =========================

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


        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
        )


    # =========================
    # 8. Load Best Model
    # =========================

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


    # =========================
    # 9. Test
    # =========================

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


    # =========================
    # 10. Predict
    # =========================

    true_labels, predictions = predict(
        model=model,
        data_loader=test_loader,
        device=device
    )


    # =========================
    # 11. Macro-F1
    # =========================

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    print(
        f"Test Macro-F1: {macro_f1:.4f}"
    )


    # =========================
    # 12. Classification Report
    # =========================

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


    # =========================
    # 13. Loss Curve
    # =========================

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
    plt.title("ResNet18 Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.show()


    # =========================
    # 14. Accuracy Curve
    # =========================

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
    plt.title("ResNet18 Training and Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.show()


    # =========================
    # 15. Confusion Matrix
    # =========================

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

    plt.title("ResNet18 Confusion Matrix")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()