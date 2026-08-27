import torch
import torch.nn as nn
from sympy.sets.sets import set_function


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        #ReLU
        self.relu1 = nn.ReLU()

        #Pooling layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )

        #[64x1x28x28]->[64x32x28x28]->ReLU->[64x32x14x14]

        # Convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # ReLU
        self.relu2 = nn.ReLU()

        # Pooling layer
        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )
        # [64x32x14x14]->[64x64x14x14]->ReLU->[64x64x7x7]

        #Flatten
        self.flatten = nn.Flatten()

        #FC layer
        self.fc1 = nn.Linear(
            64*7*7,
            128
        )

        #[64x64x7x7]->Flatten->[64x3136]->FC->[64x128]->FC->[64x11]

        #ReLU
        self.relu3 = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(
            p=0.5
        )

        #Classification layer
        self.fc2 = nn.Linear(
            128,
            11
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.dropout(x)

        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN()

    x = torch.randn(64, 1, 28, 28)

    output = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", output.shape)