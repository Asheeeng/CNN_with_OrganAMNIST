import torch

from data import train_loader
from model import TransferResNet18


# ==========================
# 1. 创建模型
# ==========================

model = TransferResNet18()


# ==========================
# 2. 获取一个batch数据
# ==========================

images, labels = next(iter(train_loader))


print("input shape:")
print(images.shape)


# ==========================
# 3. forward
# ==========================

output = model(images)


print("output shape:")
print(output.shape)