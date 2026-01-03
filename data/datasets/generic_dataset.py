from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp
from collections import Counter
import torch

from utils import listdir_nohidden, logger

kaggle_root_dir = {
    "CNNSpotTest": "/kaggle/input/cnnspot/cnn_spot/test",
    "CLIPpingEval": "/kaggle/input/clipping-eval/deepfake_eval",
}

kaggle_dataset_paths = {
    "BigGAN": osp.join(kaggle_root_dir["CNNSpotTest"], "biggan"),
    "CRN": osp.join(kaggle_root_dir["CNNSpotTest"], "crn"),
    "CycleGAN": osp.join(kaggle_root_dir["CNNSpotTest"], "cyclegan"),
    "DeepFake": osp.join(kaggle_root_dir["CNNSpotTest"], "deepfake"),
    "GauGAN": osp.join(kaggle_root_dir["CNNSpotTest"], "gaugan"),
    "IMLE": osp.join(kaggle_root_dir["CNNSpotTest"], "imle"),
    "ProGAN": osp.join(kaggle_root_dir["CNNSpotTest"], "progan"),
    "SAN": osp.join(kaggle_root_dir["CNNSpotTest"], "san"),
    "SeeingDark": osp.join(kaggle_root_dir["CNNSpotTest"], "seeingdark"),
    "StarGAN": osp.join(kaggle_root_dir["CNNSpotTest"], "stargan"),
    "StyleGAN": osp.join(kaggle_root_dir["CNNSpotTest"], "stylegan"),
    "StyleGAN2": osp.join(kaggle_root_dir["CNNSpotTest"], "stylegan2"),
    "WhichFaceIsReal": osp.join(kaggle_root_dir["CNNSpotTest"], "whichfaceisreal"),
}

class MyImageDataset(Dataset):
    def __init__(self, img_dir, split, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.split = split

        self.dataset_name = os.path.basename(self.img_dir) # each dataset has its own way of reading files
        logger.debug(f"Folder dataset name loaded is {self.dataset_name}")

        self.read_data_dir(self.split)

        self.log_dataset_info()

    def read_data_dir(self, split="test"):
        """
        Load image file paths and labels from a directory structured as:

            img_dir/
                class_name_1/     ← e.g., "cat"
                    labelName1/   ← e.g., "0_real"
                        img1.jpg
                        img2.jpg
                    labelName2/   ← e.g., "1_fake"
                        img3.jpg
                class_name_2/
                    ...
        """
        
        self.img_files = []
        self.labels = []

        if split == "test":
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
        
        return

    def log_dataset_info(self):
        total_images = len(self.img_files)
        unique_classes = set(self.labels)
        num_classes = len(unique_classes)

        counts = Counter(self.labels)

        logger.info(f"Dataset '{self.dataset_name}' loaded.")
        logger.info(f"Split: {self.split}")
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

        label = torch.tensor(label, dtype=torch.long)

        return image, label

class CNNSpot(MyImageDataset):
    def __init__(self, img_dir, split, transform=None, use_jsd=False):
        self.use_jsd = use_jsd
        super().__init__(img_dir, split, transform)
        if self.use_jsd and self.split == "train":
            self.aug, self.preprocess = self.transform
        
    def read_data_dir(self, split="test"):
        self.img_files = []
        self.labels = []

        data_dir = osp.join(self.img_dir, split)
        if split == "train" or split == "val":
            class_names = listdir_nohidden(data_dir)
            for class_name in class_names:
                class_dir = osp.join(data_dir, class_name)
                label_names = listdir_nohidden(class_dir)
                for label_name in label_names:
                    label_dir = osp.join(class_dir, label_name)
                    label = int(label_name.split("_")[0])
                    imnames = listdir_nohidden(label_dir)
                    for imname in imnames[:6000]:
                        impath = osp.join(label_dir, imname)
                        self.img_files.append(impath)
                        self.labels.append(label)
        elif split == "test":
            model_names = listdir_nohidden(data_dir)
            for model_name in model_names:
                model_dir = osp.join(data_dir, model_name)
                if model_name not in ["cyclegan", "progan", "stylegan", "stylegan2"]:
                    label_names = listdir_nohidden(model_dir)
                    for label_name in label_names:
                        label_dir = osp.join(model_dir, label_name)
                        label = int(label_name.split("_")[0])
                        imnames = listdir_nohidden(label_dir)
                        for imname in imnames:
                            impath = osp.join(label_dir, imname)
                            self.img_files.append(impath)
                            self.labels.append(label)
                else:
                    class_names = listdir_nohidden(model_dir)
                    for class_name in class_names:
                        class_dir = osp.join(model_dir, class_name)
                        label_names = listdir_nohidden(class_dir)
                        for label_name in label_names:
                            label_dir = osp.join(class_dir, label_name)
                            label = int(label_name.split("_")[0])
                            imnames = listdir_nohidden(label_dir)
                            for imname in imnames:
                                impath = osp.join(label_dir, imname)
                                self.img_files.append(impath)
                                self.labels.append(label)
        
        return

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.use_jsd and self.split == "train":
            im_tuple = (self.preprocess(image), self.aug(image), self.aug(image))
            return im_tuple, label
        
        if self.transform:
            image = self.transform(image)

        return image, label

class CNNSpotTestSet(MyImageDataset):
    def read_data_dir(self, split="test"):
        self.img_files = []
        self.labels = []

        data_dir = kaggle_dataset_paths[self.dataset_name]
        if split == "test":
            if self.dataset_name.lower() not in ["cyclegan", "progan", "stylegan", "stylegan2"]:
                label_names = listdir_nohidden(data_dir)
                for label_name in label_names:
                    label_dir = osp.join(data_dir, label_name)
                    label = int(label_name.split("_")[0])
                    imnames = listdir_nohidden(label_dir)
                    for imname in imnames:
                        impath = osp.join(label_dir, imname)
                        self.img_files.append(impath)
                        self.labels.append(label)
            else:
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
    
        return
