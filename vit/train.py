import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import torch.nn.functional as F
from functools import partial
import torchmetrics
from utils import linear_warmup_cosine_decay, ssl_quantize
from vit import VisionTransformer
from PIL import Image
import os
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

# dagshub.init("Medical-MNIST-MLOPs-CT-CD", "JatinSingh28", mlflow=True)
# mlflow.start_run()


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, transformer=None, debug=False):
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
        print(img.size)

        if self.transformer:
            img = self.transformer(img)
        return img, label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class ViT(pl.LightningModule):
    def __init__(
        self,
        steps=2,
        learning_rate=1e-4,
        weight_decay=0.0001,
        image_size=224,
        num_classes=6,
        patch_size=4,
        dim=256,
        layers=12,
        heads=8,
        dropout_p=0.0,
        linear_warmup_ratio=0.05,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
        **_,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio

        self.model = VisionTransformer(
            image_size=image_size,
            num_classes=num_classes,
            patch_size=patch_size,
            dim=dim,
            layers=layers,
            heads=heads,
            dropout_p=dropout_p,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        #self.val_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        warmup_steps = int(self.linear_warmup_ratio * self.steps)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, linear_warmup_cosine_decay(warmup_steps, self.steps)
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.val_accuracy(y_hat, y)
        return loss

    #def validation_epoch_end(self, losses):
    #    self.log("valid_loss", torch.stack(losses).mean(), prog_bar=True)
    #    self.log("valid_acc", self.val_accuracy.compute(), prog_bar=True)


dataset = DataModule(data_dir="./data", batch_size=32, transformer=transform, debug=True)

train_size = int(0.8 * dataset.__len__())
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    dirpath="model_ckpt",
    filename="sample-mnist-epoch{epoch:02d}-val_loss{val_loss:.2f}",
    auto_insert_metric_name=False,
    save_top_k=3,
    save_last=True,
    save_weights_only=True,
)

pl.seed_everything(42)
model = ViT(steps=1)
trainer = pl.Trainer(max_epochs=1, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)

# mlflow.end_run()
