from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import (
    RandomApply, Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, GaussianBlur
)
from torchvision.transforms.v2 import JPEG, GaussianNoise
import numpy as np
import torch

from models import MODEL_TO_BACKBONES

from utils import logger

AVAI_CHOICES = [
    "resize",
    "random_flip",
    "gaussian_blur",
    "jpeg_compression",
    # add more corruptions here
    "to_tensor",
    "gaussian_noise", # Gaussian Noise does not support PIL images
    "normalize",
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

class BackbonePreprocessWrapper:
    def __init__(self, backbones):
        self.backbones = backbones

    def __call__(self, x):
        return [backbone.preprocess(x) for backbone in self.backbones]
    
class VisualizePreprocessWrapper:
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img, dtype=np.float32) / 255.0 # HWC, float32
        img = torch.tensor(img).permute(2, 0, 1) # convert to tensor CHW
        return img

def build_model_transform(cfg):
    if cfg.MODEL.NAME in MODEL_TO_BACKBONES:
        logger.info("+ normalize by backbone")
        return BackbonePreprocessWrapper(MODEL_TO_BACKBONES[cfg.MODEL.NAME])
    else:
        logger.error(f"Unknown model name: {cfg.MODEL.NAME}")
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")

def build_transform(cfg, is_train, is_visualize=False, use_jsd=False):
    # In case JSD loss is used, a preprocess (no augmentation) transform should be returned
    if is_visualize:
        return VisualizePreprocessWrapper()

    choices = cfg.TRANSFORM.TRANSFORMS

    for choice in choices:
        if choice not in AVAI_CHOICES:
            logger.error(f"{choice} is not in {AVAI_CHOICES}")
            raise ValueError(f"{choice} is not in {AVAI_CHOICES}")
        
    if is_train:
        logger.info("Building transform train...")
    else:
        logger.info("Building transform val/test...")

    tfm = []
    preprocess = []

    interp_mode = INTERPOLATION_MODES[cfg.TRANSFORM.INTERPOLATION_MODE]
    input_size = cfg.TRANSFORM.INPUT_SIZE

    if "resize" in choices:
        logger.info(f"+ resize to {input_size}")
        tfm += [Resize(input_size, interpolation=interp_mode)]
        preprocess += [Resize(input_size, interpolation=interp_mode)]

    if is_train or not cfg.TRANSFORM.NO_TRANSFORM_TEST:
        if "random_flip" in choices:
            logger.info("+ random flip")
            tfm += [RandomHorizontalFlip()]
    
        if "gaussian_blur" in choices:
            logger.info(f"+ gaussian blur (p={cfg.TRANSFORM.GB_P}, kernel={cfg.TRANSFORM.GB_K}, sigma={cfg.TRANSFORM.GB_SIGMA})")
            gb_k, gb_p, gb_sigma = cfg.TRANSFORM.GB_K, cfg.TRANSFORM.GB_P, cfg.TRANSFORM.GB_SIGMA
            tfm += [RandomApply([GaussianBlur(gb_k, gb_sigma)], p=gb_p)]

        if "jpeg_compression" in choices:
            logger.info(f"+ jpeg compression (p={cfg.TRANSFORM.JPEG_P}, quality={cfg.TRANSFORM.JPEG_QUALITY})")
            jpeg_p, jpeg_quality = cfg.TRANSFORM.JPEG_P, cfg.TRANSFORM.JPEG_QUALITY
            tfm += [RandomApply([JPEG(quality=jpeg_quality)], p=jpeg_p)]

    if "to_tensor" in choices:
        logger.info("+ to torch tensor of range [0, 1]")
        tfm += [ToTensor()]
        preprocess += [ToTensor()]

    # Gaussian noise MUST be after ToTensor
    if is_train or not cfg.TRANSFORM.NO_TRANSFORM_TEST:
        if "to_tensor" in choices and "gaussian_noise" in choices:
            logger.info(
                f"+ gaussian noise (p={cfg.TRANSFORM.GN_P}, "
                f"mean={cfg.TRANSFORM.GN_MEAN}, sigma={cfg.TRANSFORM.GN_SIGMA})"
            )
            tfm += [
                RandomApply(
                    [GaussianNoise(
                        mean=cfg.TRANSFORM.GN_MEAN,
                        sigma=cfg.TRANSFORM.GN_SIGMA,
                        clip=True
                    )],
                    p=cfg.TRANSFORM.GN_P
                )
            ]

    if "normalize" in choices:
        if cfg.TRANSFORM.NORMALIZE_BACKBONE:
            tfm += [build_model_transform(cfg)]
            preprocess += [build_model_transform(cfg)]
        else:
            logger.info(
                f"+ normalization (mean={cfg.TRANSFORM.INPUT_MEAN}, std={cfg.TRANSFORM.INPUT_STD})"
            )
            tfm += [Normalize(mean=cfg.TRANSFORM.INPUT_MEAN, std=cfg.TRANSFORM.INPUT_STD)]
            preprocess += [Normalize(mean=cfg.TRANSFORM.INPUT_MEAN, std=cfg.TRANSFORM.INPUT_STD)]

    tfm = Compose(tfm)
    preprocess = Compose(preprocess)
    
    if use_jsd:
        return tfm, preprocess
    
    return tfm
