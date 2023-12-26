import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

device = torch.device("cuda:0")

transforms = transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transforms,
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transforms,
    download=True,
)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8)


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = CIFAR10()
net.to(device)
print(net)


criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

epochs = 8

writer = SummaryWriter("./runs")

for epoch in range(epochs):
    print(f"epoch: [{epoch + 1}/{epochs}]")

    total_train_loss = 0
    total_test_loss = 0
    total_train_step = 0
    total_test_step = 0
    total_train_accuracy = 0
    total_test_accuracy = 0

    net.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        pred_targets = net(imgs)
        accuracy = (pred_targets.argmax(1) == targets).sum()
        total_train_accuracy += accuracy

        loss = criterion(pred_targets, targets)
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

    avg_train_loss = total_train_loss / len(train_data)
    avg_train_accuracy = total_train_accuracy / len(train_data)
    print(
        "train_loss: [%.4f] | train_accuracy: [%.4f]"
        % (avg_train_loss, avg_train_accuracy)
    )
    writer.add_scalar("train_loss", avg_train_loss, epoch)
    writer.add_scalar("train_accuracy", avg_train_accuracy, epoch)

    net.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            pred_targets = net(imgs)
            accuracy = (pred_targets.argmax(1) == targets).sum()
            total_test_accuracy += accuracy

            loss = criterion(pred_targets, targets)
            total_test_loss += loss.item()

            total_test_step += 1

        avg_test_loss = total_test_loss / len(test_data)
        avg_test_accuracy = total_test_accuracy / len(test_data)
        print(
            "test_loss: [%.4f] | test_accuracy: [%.4f]"
            % (avg_test_loss, avg_test_accuracy)
        )
        writer.add_scalar("test_loss", avg_test_loss, epoch)
        writer.add_scalar("test_accuracy", avg_test_accuracy, epoch)
writer.close()

net_save_path = "./checkpoints"
if not os.path.exists(net_save_path):
    os.makedirs(net_save_path)

torch.save(net.state_dict(), "./checkpoints/net_last.pth")
