import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        # =========================
        # 主分支 F(x)
        # =========================

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )


        # =========================
        # Shortcut
        # =========================

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()


    def forward(self, x):

        # shortcut 分支
        identity = self.shortcut(x)

        # 主分支 F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # F(x) + x
        out = out + identity

        out = self.relu(out)

        return out


if __name__ == "__main__":

    # 情况1：shape 不变

    x = torch.randn(
        64,
        64,
        28,
        28
    )

    block1 = BasicBlock(
        in_channels=64,
        out_channels=64,
        stride=1
    )

    y = block1(x)

    print("情况1")
    print("输入:", x.shape)
    print("输出:", y.shape)


    # 情况2：下采样 + channel 增加

    x = torch.randn(
        64,
        64,
        28,
        28
    )

    block2 = BasicBlock(
        in_channels=64,
        out_channels=128,
        stride=2
    )

    y = block2(x)

    print()
    print("情况2")
    print("输入:", x.shape)
    print("输出:", y.shape)