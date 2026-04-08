from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import (
    RandomApply, Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, GaussianBlur, ColorJitter
)
from torchvision.transforms.v2 import JPEG, GaussianNoise
import numpy as np
import torch

import cv2 
from scipy.signal import wiener
import pywt
from skimage.restoration import denoise_wavelet
# Gaussian Blur: Kernel size (k, k), Sigma (σ)
# (3,3), sigma=0.5
# (5,5), sigma=1.0
# (7,7), sigma=1.5
# (9,9), sigma=2.0
# Wiener filter: Window size (m, n), (Optional) Noise variance (if manually set)
# (3,3)
# (5,5)
# (7,7)
# (9,9)
# Wavelet denoising: wavelet type, wavelet_levels, method (BayesShrink / VisuShrink), mode (soft / hard thresholding)
# denoise_wavelet(img,
#                 wavelet='db1',
#                 wavelet_levels=2,
#                 method='BayesShrink',
#                 mode='soft')
# 'db1', 'db2', 'sym4', 'coif1'
# levels = 1, 2, 3, 4
# 'BayesShrink' → ✅ best general choice
# 'VisuShrink' → stronger denoising but may oversmooth
# 'soft' → smoother (recommended)
# 'hard' → sharper but can look unnatural

from PIL import Image

from models import MODEL_TO_BACKBONES

from utils import logger
from utils import RandomNTIREDistortion

AVAI_CHOICES = [
    "resize",
    "random_flip",
    "brightness",
    "contrast",
    "gaussian_blur",
    "wiener_filter",
    "wavelet_denoising",
    "jpeg_compression",
    # add more corruptions here
    "to_tensor",
    "random_ntire_distortion",
    "gaussian_noise", # Gaussian Noise does not support PIL images
    "normalize",
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

class WienerFilterTransform:
    def __init__(self, mysize=(5, 5), noise=None):
        self.mysize = mysize
        self.noise = noise

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)

        if img_np.ndim == 3:
            out = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                out[..., c] = wiener(img_np[..., c], mysize=self.mysize, noise=self.noise)
        else:
            out = wiener(img_np, mysize=self.mysize, noise=self.noise)

        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    
# JSD? PIL?

class WaveletDenoiseTransform:
    def __init__(self, wavelet='db1', wavelet_levels=2,
                 method='BayesShrink', mode='soft'):
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.method = method
        self.mode = mode

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0

        out = denoise_wavelet(
            img_np,
            wavelet=self.wavelet,
            wavelet_levels=self.wavelet_levels,
            method=self.method,
            mode=self.mode,
            channel_axis=-1 if img_np.ndim == 3 else None
        )

        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

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

        if "brightness" in choices:
            logger.info(f"+ brightness (p={cfg.TRANSFORM.BRIGHTNESS_P}, brightness={cfg.TRANSFORM.BRIGHTNESS})")
            brightness, brightness_p = cfg.TRANSFORM.BRIGHTNESS, cfg.TRANSFORM.BRIGHTNESS_P
            tfm += [RandomApply([ColorJitter(brightness=brightness)], p=brightness_p)]

        if "contrast" in choices:
            logger.info(f"+ contrast (p={cfg.TRANSFORM.CONTRAST_P}, contrast={cfg.TRANSFORM.CONTRAST})")
            contrast, contrast_p = cfg.TRANSFORM.CONTRAST, cfg.TRANSFORM.CONTRAST_P
            tfm += [RandomApply([ColorJitter(contrast=contrast)], p=contrast_p)]
    
        if "gaussian_blur" in choices:
            logger.info(f"+ gaussian blur (p={cfg.TRANSFORM.GB_P}, kernel={cfg.TRANSFORM.GB_K}, sigma={cfg.TRANSFORM.GB_SIGMA})")
            gb_k, gb_p, gb_sigma = cfg.TRANSFORM.GB_K, cfg.TRANSFORM.GB_P, cfg.TRANSFORM.GB_SIGMA
            tfm += [RandomApply([GaussianBlur(gb_k, gb_sigma)], p=gb_p)]

        if "wiener_filter" in choices:
            logger.info(
                f"+ wiener filter (p={cfg.TRANSFORM.WIENER_P}, "
                f"kernel={cfg.TRANSFORM.WIENER_K}, noise={cfg.TRANSFORM.WIENER_NOISE})"
            )
            tfm += [
                RandomApply(
                    [WienerFilterTransform(
                        mysize=cfg.TRANSFORM.WIENER_K,
                        noise=cfg.TRANSFORM.WIENER_NOISE
                    )],
                    p=cfg.TRANSFORM.WIENER_P
                )
            ]

        if "wavelet_denoising" in choices:
            logger.info(
                f"+ wavelet denoising (p={cfg.TRANSFORM.WAVELET_P}, "
                f"wavelet={cfg.TRANSFORM.WAVELET_TYPE}, "
                f"levels={cfg.TRANSFORM.WAVELET_LEVELS})"
            )
            tfm += [
                RandomApply(
                    [WaveletDenoiseTransform(
                        wavelet=cfg.TRANSFORM.WAVELET_TYPE,
                        wavelet_levels=cfg.TRANSFORM.WAVELET_LEVELS,
                        method=cfg.TRANSFORM.WAVELET_METHOD,
                        mode=cfg.TRANSFORM.WAVELET_MODE
                    )],
                    p=cfg.TRANSFORM.WAVELET_P
                )
            ]

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
        if "to_tensor" in choices and "random_ntire_distortion" in choices:
            tf = RandomNTIREDistortion()
            logger.info(f"+ {tf}")
            tfm += [
                tf
            ]

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
            backbone_norm = build_model_transform(cfg)
            tfm += [backbone_norm]
            preprocess += [backbone_norm]
        else:
            logger.info(
                f"+ normalization (mean={cfg.TRANSFORM.INPUT_MEAN}, std={cfg.TRANSFORM.INPUT_STD})"
            )
            tfm += [Normalize(mean=cfg.TRANSFORM.INPUT_MEAN, std=cfg.TRANSFORM.INPUT_STD)]
            preprocess += [Normalize(mean=cfg.TRANSFORM.INPUT_MEAN, std=cfg.TRANSFORM.INPUT_STD)]

    tfm = Compose(tfm)
    preprocess = Compose(preprocess)
    
    if use_jsd and is_train:
        return tfm, preprocess
    
    return tfm
