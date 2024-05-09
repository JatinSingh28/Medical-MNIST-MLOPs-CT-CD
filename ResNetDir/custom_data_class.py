import pytorch_lightning as pl
import os
from PIL import Image

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