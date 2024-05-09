import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import mlflow


class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=6):
        super(ResNetModel, self).__init__()
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.model = resnet

    def forward(self, x):
        return self.model(x)

    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     # Adjust the input size of the first fully connected layer
    #     self.fc1 = nn.Linear(16 * 13 * 13, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, num_classes)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     # Adjust the flatten function to accommodate different spatial dimensions
    #     x = flatten(x, 1)  # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    #     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.conv3 = nn.Conv2d(64, 128, 3, 1)
    #     self.fc1 = nn.Linear(128 * 8 * 8, 512)
    #     self.fc2 = nn.Linear(512, num_classes)
    #     self.pool = nn.MaxPool2d(2, 2)

    # def forward(self, x):
    #     x = F.relu(self.pool(self.conv1(x), 2))
    #     x = F.relu(self.pool(self.conv2(x), 2))
    #     x = F.relu(self.pool(self.conv3(x), 2))
    #     x = flatten(x,1)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train loss", loss)
        mlflow.log_metric("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        mlflow.log_metric("val_loss", loss)
        # return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def test(self, dataset):
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.eval()
        test_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(loader)
        mlflow.log_metric("test_loss", test_loss)
        return test_loss
