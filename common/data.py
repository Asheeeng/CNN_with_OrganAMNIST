import numpy as np

from torch.utils.data import DataLoader, Subset
from medmnist import OrganAMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    batch_size=64,
    samples_per_class=None,
    seed=42
):

    # =========================
    # 1. Transform
    # =========================

    transform = transforms.ToTensor()


    # =========================
    # 2. Dataset
    # =========================

    train_dataset = OrganAMNIST(
        split="train",
        transform=transform,
        download=True
    )

    if samples_per_class is not None:

        labels = train_dataset.labels.squeeze()

        rng = np.random.default_rng(seed)

        selected_indices = []

        for class_id in range(11):
            class_indices = np.where(
                labels == class_id
            )[0]

            sampled_indices = rng.choice(
                class_indices,
                size=samples_per_class,
                replace=False
            )

            selected_indices.extend(
                sampled_indices.tolist()
            )

        train_dataset = Subset(
            train_dataset,
            selected_indices
        )

        

    val_dataset = OrganAMNIST(
        split="val",
        transform=transform,
        download=True
    )

    test_dataset = OrganAMNIST(
        split="test",
        transform=transform,
        download=True
    )


    # =========================
    # 3. DataLoader
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # =========================
    # 4. 打印数据量
    # =========================

    print("训练集:", len(train_dataset))
    print("验证集:", len(val_dataset))
    print("测试集:", len(test_dataset))


    # =========================
    # 5. 返回 DataLoader
    # =========================

    return train_loader, val_loader, test_loader