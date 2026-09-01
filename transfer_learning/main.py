import torch
import torch.nn as nn

from data import (
    train_loader,
    val_loader,
    test_loader
)

from model import TransferResNet18



# ======================
# device
# ======================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ======================
# model
# ======================

model = TransferResNet18()

model = model.to(device)



# ======================
# loss
# ======================

criterion = nn.CrossEntropyLoss()



# ======================
# optimizer
# ======================

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    #full->5e-6  other:1e-3
    lr=5e-6
)



# ======================
# train
# ======================

def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for images, labels in train_loader:


        images = images.to(device)

        labels = labels.squeeze(1)
        labels = labels.long().to(device)



        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()



        total_loss += loss.item()


        pred = outputs.argmax(dim=1)


        correct += (
            pred == labels
        ).sum().item()


        total += labels.size(0)



    avg_loss = total_loss / len(train_loader)

    acc = correct / total


    return avg_loss, acc





# ======================
# validation
# ======================


def evaluate(loader):

    model.eval()


    total_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in loader:


            images = images.to(device)


            labels = labels.squeeze(1)
            labels = labels.long().to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            total_loss += loss.item()


            pred = outputs.argmax(dim=1)


            correct += (
                pred == labels
            ).sum().item()


            total += labels.size(0)



    avg_loss = total_loss / len(loader)

    acc = correct / total


    return avg_loss, acc





# ======================
# training loop
# ======================


epochs = 10


best_val_acc = 0



for epoch in range(epochs):


    train_loss, train_acc = train_one_epoch()


    val_loss, val_acc = evaluate(
        val_loader
    )


    print(
        f"""
Epoch [{epoch+1}/{epochs}]

Train Loss: {train_loss:.4f}
Train Acc : {train_acc:.4f}

Val Loss  : {val_loss:.4f}
Val Acc   : {val_acc:.4f}

"""
    )



    if val_acc > best_val_acc:

        best_val_acc = val_acc


        torch.save(
            model.state_dict(),
            "best_transfer_resnet18.pth"
        )


print(
    "Best Val Acc:",
    best_val_acc
)



# ======================
# test
# ======================


model.load_state_dict(
    torch.load(
        "best_transfer_resnet18.pth"
    )
)


test_loss, test_acc = evaluate(
    test_loader
)


print(
    "Test Loss:",
    test_loss
)

print(
    "Test Acc:",
    test_acc
)