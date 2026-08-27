import torch


def train_one_epoch(
        model,
        data_loader,
        criterion,
        optimizer,
        device
):
    # 训练模式，Dropout 开启
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)

        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(images)

        # 3. 计算 loss
        loss = criterion(outputs, labels)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # -------------------------
        # 统计结果
        # -------------------------

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(
        model,
        data_loader,
        criterion,
        device
):
    # 评估模式，Dropout 关闭
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.squeeze(1).long().to(device)

            # 只做前向传播
            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def predict(
        model,
        data_loader,
        device
):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.squeeze(1).long().to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    return all_labels, all_predictions