from medmnist import OrganAMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor()

])


train_dataset = OrganAMNIST(
    split="train",
    transform=transform,
    download=True
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


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


if __name__ == "__main__":

    images, labels = next(iter(train_loader))

    print("images shape:", images.shape)
    print("labels shape:", labels.shape)