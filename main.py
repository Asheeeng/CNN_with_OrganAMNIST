import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import SimpleCNN
from data import get_dataloaders
from engine import train_one_epoch, evaluate, predict

'''
SimpleCNN
 Train / Val / Test
 Dropout
 Best checkpoint
 Loss / Accuracy curve
 模块化
 Macro-F1
 Confusion Matrix
'''


def main():

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
    else:
        device = torch.device("cpu")

    print("使用设备:", device)


    # =========================
    # 3. Model
    # =========================

    model = SimpleCNN().to(device)


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

        # 保存曲线数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                "best_model.pth"
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
            "best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    print(
        f"\n已加载最佳模型：Epoch {best_epoch}, "
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

    from sklearn.metrics import (
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay
    )

    true_labels, predictions = predict(
        model=model,
        data_loader=test_loader,
        device=device
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    print(
        f"Test Macro-F1: {macro_f1:.4f}"
    )

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

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # =========================
    # 10. Plot Loss
    # =========================

    epochs = range(1, num_epochs + 1)

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
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.show()


    # =========================
    # 11. Plot Accuracy
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
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()