import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class TransferResNet18(nn.Module):

    def __init__(
            self,
            mode = "full",
        ):

        super().__init__()


        # 1. pretrained ResNet18

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )


        # 2. 替换分类头

        self.model.fc = nn.Linear(
            512,
            11
        )


        # 3. 冻结backbone

        if mode == "freeze":

            for name, param in self.model.named_parameters():

                if "fc" not in name:
                    param.requires_grad = False



        elif mode == "partial":

            for name, param in self.model.named_parameters():

                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False



        elif mode == "full":

            for param in self.model.parameters():
                param.requires_grad = True



    def forward(self,x):

        return self.model(x)