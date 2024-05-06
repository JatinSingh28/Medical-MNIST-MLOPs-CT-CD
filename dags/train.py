import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from mlflow import MlflowClient
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
# import os
import mlflow
from dotenv import load_dotenv
import dagshub
import sys

sys.path.append("../")
from ResNetDir.initial_train import DataModule
# from ResNet.initial_train import DataModule

def continious_train():
    load_dotenv()
    dagshub.init("Medical-MNIST-MLOPs-CT-CD", "JatinSingh28", mlflow=True)
    mlflow.start_run()
    mlflow.autolog()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

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

    dataset = DataModule(data_dir="./AWS_data", transformer=transform)
    
    train_size = int(0.8 * dataset.__len__())
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    pl.seed_everything(42)
    # model = ResNetModel()

    client = MlflowClient()
    registered_model_name = "Production Model V1"
    version = client.get_model_version_by_alias(registered_model_name,"prod").version
    # version = client.get_latest_versions(name=registered_model_name)[0].version

    model_uri = f'models:/{registered_model_name}/{version}'
    
    model = mlflow.pytorch.load_model(model_uri)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath="model_ckpt",
        filename="mnist-epoch{epoch:02d}-val_loss{val_loss:.5f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        save_last=True,
        save_weights_only=True,
    )

    trainer = pl.Trainer(max_epochs=3, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    loss = model.test(dataset)
    print(loss)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model_artifact",
        registered_model_name="Production Model V1",
        metadata={"Val Loss": loss},
    )

    version = client.get_latest_versions("Production Model V1")[0].version
    client.set_registered_model_alias("Production Model V1", "prod", version)

    mlflow.end_run()

if __name__ == "__main__":
    continious_train()
