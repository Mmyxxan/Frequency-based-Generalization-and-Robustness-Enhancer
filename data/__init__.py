from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from datasets import load_dataset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import torch

from .datasets.generic_dataset import *
from .datasets.hugging_face_dataset import HuggingFaceIterableDataset
from .transforms.generic_transform import build_transform

from utils import logger

def build_dataset(cfg, is_train, split, is_visualize=False, transform=None):
    if not transform:
        transform = build_transform(cfg=cfg, is_train=is_train, is_visualize=is_visualize, use_jsd=cfg.RoHL.USE_JSD)

    if cfg.DATASET.NAME == "myxxanaplt/TrueFake-647GB":
        logger.info(f"Loading dataset {cfg.DATASET.NAME} from HuggingFace...")

        ds = load_dataset("myxxanaplt/TrueFake-647GB", split=split, streaming=True)
        
        return HuggingFaceIterableDataset(hf_dataset=ds, transform=transform) # haven't implement JSD yet, but it's ok because Truefake is not used for training
    else:
        logger.info(f"Loading dataset {cfg.DATASET.NAME} locally...")
    
    if cfg.DATASET.NAME == "CNNSpot":
        return CNNSpot(img_dir=cfg.DATASET.DATA_DIR, split=split, transform=transform, use_jsd=cfg.RoHL.USE_JSD)
    elif cfg.DATASET.NAME == "CNNSpotTestSet":
        return CNNSpotTestSet(img_dir=cfg.DATASET.DATA_DIR, split=split, transform=transform)
    elif cfg.DATASET.NAME == "NTIRE2026Dataset":
        return NTIRE2026Dataset(img_dir=cfg.DATASET.DATA_DIR, split=split, transform=transform, use_jsd=cfg.RoHL.USE_JSD, shard_dirs=["/kaggle/input/datasets/anhphmminh/ntiretrainingset-shard0/shard_0", "/kaggle/input/datasets/anhphmminh/ntiretrainingset-shard1/shard_1", "/kaggle/input/datasets/anhphmminh/ntiretrainingset-shard2/shard_2", "/kaggle/input/datasets/anhphmminh/ntiretrainingset-shard3/shard_3", "/kaggle/input/datasets/anhphmminh/ntiretrainingset-shard4/shard_4", "/kaggle/input/datasets/anhphmminh/ntiretrainingset-shard5/shard_5"])
    elif cfg.DATASET.NAME == "UniversalTrainingSet":
        return UniversalTrainingSet(img_dir=cfg.DATASET.DATA_DIR, split=split, transform=transform, use_jsd=cfg.RoHL.USE_JSD)
    elif cfg.DATASET.NAME == "ReconstructedFakeRealDataset":
        return ReconstructedFakeRealDataset(img_dir=cfg.DATASET.DATA_DIR, split=split, transform=transform, use_jsd=cfg.RoHL.USE_JSD, edit_types=cfg.DATASET.RECONSTRUCTED_FAKE_REAL_DATASET.EDIT_TYPES)
    else:
        return MyImageDataset(img_dir=cfg.DATASET.DATA_DIR, split=split, transform=transform)

def build_dataloader(cfg, is_train, split, is_visualize=False, transform=None):
    dataset = build_dataset(cfg=cfg, is_train=is_train, split=split, is_visualize=is_visualize, transform=transform)
    is_iterable = isinstance(dataset, IterableDataset)
    
    # WeightedRandomSampler for imbalanced dataset
    labels = dataset.labels

    class_count = Counter(labels)
    print(class_count)

    # inverse frequency
    class_weights = {
        c: 1.0 / n
        for c, n in class_count.items()
    }

    sample_weights = torch.DoubleTensor(
        [class_weights[l] for l in labels]
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    if is_train and not is_iterable:
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True
        )

        return DataLoader(
            dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            sampler=sampler,
            shuffle=False,      # sampler and shuffle cannot coexist
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=len(dataset) >= cfg.DATALOADER.BATCH_SIZE,
            pin_memory=cfg.TRAINER.USE_CUDA,
        )

    return DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=not is_iterable and is_train,  # can't shuffle IterableDataset
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=not is_iterable and is_train and len(dataset) >= cfg.DATALOADER.BATCH_SIZE,
        pin_memory=cfg.TRAINER.USE_CUDA,
    )
