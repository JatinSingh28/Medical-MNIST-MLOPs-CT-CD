import torch.nn as nn

import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

# from lightning.pytorch.loggers import MLFlowLogger
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import os

import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

dagshub.init("Medical-Image-Classification", "JatinSingh28", mlflow=True)
mlflow.start_run()


class DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir="../data", batch_size=32, transformer=None, debug=False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transformer = transformer
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_id = {class_name: i for i, class_name in enumerate(self.classes)}
        self.image_path = self.get_img_path()
        if debug:
            self.image_path = self.image_path[:100]

    def get_img_path(self):
        img_paths = []
        for classs in self.classes:
            class_dir = os.path.join(self.data_dir, classs)
            img_names = os.listdir(class_dir)
            for img in img_names:
                img_paths.append(
                    (os.path.join(class_dir, img), self.class_to_id[classs])
                )

        return img_paths

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path, label = self.image_path[idx]
        img = Image.open(img_path)

        if self.transformer:
            img = self.transformer(img)

        return img, label


transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class CNNModel(pl.LightningModule):
    def __init__(self, num_classes=6):
        super(CNNModel, self).__init__()
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


mlflow.log_params(
    {
        "batch_size": 32,
        "epochs": 3,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "loss": "CrossEntropy",
        "model": "ResNet18",
        "debug": False,
    }
)
dataset = DataModule(transformer=transform)

train_size = int(0.8 * dataset.__len__())
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, num_workers=10, batch_size=32, prefetch_factor=2, shuffle=True
)
val_loader = DataLoader(val_dataset, num_workers=10, batch_size=32, prefetch_factor=2)

pl.seed_everything(42)
model = CNNModel()

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    dirpath="model_ckpt",
    filename="mnist-epoch{epoch:02d}-val_loss{val_loss:.4f}",
    auto_insert_metric_name=False,
    save_top_k=3,
    save_last=True,
    save_weights_only=True,
)

trainer = pl.Trainer(max_epochs=3, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)

mlflow.end_run()
