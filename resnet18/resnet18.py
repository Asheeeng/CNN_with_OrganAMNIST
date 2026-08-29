import torch
import torch.nn as nn

from .basic_block import BasicBlock

class ResNet18(nn.Module):

    def __init__(self, num_classes=11):
        super().__init__()

        # 当前 feature map 的 channel 数
        self.in_channels = 64


        # =========================
        # Stem
        # =========================

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()


        # =========================
        # Stage 1
        # =========================

        self.layer1 = self._make_layer(
            out_channels=64,
            num_blocks=2,
            stride=1
        )


        # =========================
        # Stage 2
        # =========================

        self.layer2 = self._make_layer(
            out_channels=128,
            num_blocks=2,
            stride=2
        )


        # =========================
        # Stage 3
        # =========================

        self.layer3 = self._make_layer(
            out_channels=256,
            num_blocks=2,
            stride=2
        )


        # =========================
        # Stage 4
        # =========================

        self.layer4 = self._make_layer(
            out_channels=512,
            num_blocks=2,
            stride=2
        )


        # =========================
        # Classification
        # =========================

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            512,
            num_classes
        )


    def _make_layer(
        self,
        out_channels,
        num_blocks,
        stride
    ):

        layers = []


        # =========================
        # 第一个 BasicBlock
        # 可能改变 channel / feature map 尺寸
        # =========================

        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride
            )
        )


        # 更新当前 channel
        self.in_channels = out_channels


        # =========================
        # 后面的 BasicBlock
        # shape 不再变化
        # =========================

        for _ in range(1, num_blocks):

            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1
                )
            )


        return nn.Sequential(*layers)


    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 4 Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling
        x = self.avgpool(x)

        # [B, 512, 1, 1] -> [B, 512]
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x


if __name__ == "__main__":

    model = ResNet18(
        num_classes=11
    )

    x = torch.randn(
        64,
        1,
        28,
        28
    )

    output = model(x)

    print()
    print("输入 shape:", x.shape)
    print("输出 shape:", output.shape)


    # =========================
    # 参数量
    # =========================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("总参数量:", total_params)
    print("可训练参数量:", trainable_params)