from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp
from collections import Counter

from utils import listdir_nohidden

import logging
logger = logging.getLogger(__name__)

class MyImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.dataset_name = os.path.basename(self.img_dir) # each dataset has its own way of reading files

        self.read_data_dir()

        self.log_dataset_info()

    def read_data_dir(self, split="train"):
        self.img_files = []
        self.labels = []

        data_dir = osp.join(self.img_dir)
        class_names = listdir_nohidden(data_dir)
        for class_name in class_names:
            class_dir = osp.join(data_dir, class_name)
            label_names = listdir_nohidden(class_dir)
            for label_name in label_names:
                label_dir = osp.join(class_dir, label_name)
                label = int(label_name.split("_")[0])
                imnames = listdir_nohidden(label_dir)
                for imname in imnames:
                    impath = osp.join(label_dir, imname)
                    self.img_files.append(impath)
                    self.labels.append(label)

    def log_dataset_info(self):
        total_images = len(self.img_files)
        unique_classes = set(self.labels)
        num_classes = len(unique_classes)

        counts = Counter(self.labels)

        logger.info(f"Dataset '{self.dataset_name}' loaded.")
        logger.info(f"Total images: {total_images}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Number of images per class: {dict(counts)}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
