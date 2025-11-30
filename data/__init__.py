from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from datasets import load_dataset

from .datasets.generic_dataset import MyImageDataset
from .datasets.hugging_face_dataset import HuggingFaceIterableDataset
from .transforms.generic_transform import build_transform

import logging
logger = logging.getLogger(__name__)

def build_dataset(cfg, is_train):
    transform = build_transform(cfg=cfg, is_train=is_train)

    if cfg.DATASET.NAME == "myxxanaplt/TrueFake-647GB":
        logger.info(f"Loading dataset {cfg.DATASET.NAME} from HuggingFace...")

        ds = load_dataset("myxxanaplt/TrueFake-647GB", split="test", streaming=True)
        
        return HuggingFaceIterableDataset(hf_dataset=ds, transform=transform)
    else:
        logger.info(f"Loading dataset {cfg.DATASET.NAME} locally...")

        # if is_train:
        #     img_dir = osp.join(cfg.DATASET.DATA_DIR, "train")
        # else:
        #     img_dir = osp.join(cfg.DATASET.DATA_DIR, "val")

        return MyImageDataset(img_dir=cfg.DATASET.DATA_DIR, transform=transform)

def build_dataloader(cfg, is_train):
    dataset = build_dataset(cfg=cfg, is_train=is_train)
    is_iterable = isinstance(dataset, IterableDataset)

    return DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=not is_iterable and is_train,  # can't shuffle IterableDataset
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=not is_iterable and is_train and len(dataset) >= cfg.DATALOADER.BATCH_SIZE,
        pin_memory=cfg.USE_CUDA,
    )
