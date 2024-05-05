import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from ResNet.ResNet import ResNetModel
# from ResNet import ResNetModel
from PIL import Image
import os
import dagshub
import mlflow
import mlflow.pytorch

# from mlflow.models import infer_signature
from dotenv import load_dotenv
from mlflow import MlflowClient
import mlflow.pyfunc
import os


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


if __name__ == "__main__":
    load_dotenv()
    dagshub.init("Medical-Image-Classification", "JatinSingh28", mlflow=True)
    mlflow.start_run()
    mlflow.autolog()

    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    dataset = DataModule(transformer=transform)

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
    print(version)
    # version = client.get_latest_versions(name=registered_model_name)[0].version

    model_uri = f'models:/{registered_model_name}/{version}'
    print(model_uri)
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

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model_artifact",
        registered_model_name="Production Model V1",
        metadata={"Val Loss": loss},
    )

    version = client.get_latest_versions("Production Model V1")[0].version
    client.set_registered_model_alias("Production Model V1", "prod", version)

    # run_id = run.info.run_id
    # model_uri = "runs:/" + run_id + "/model"
    # registered_model_name = "TestModel"
    # experiment_name = "TestExperiment"
    # mlflow.set_experiment(experiment_name)
    # mlflow.register_model(model_uri, registered_model_name)

    # mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model")

    mlflow.end_run()
