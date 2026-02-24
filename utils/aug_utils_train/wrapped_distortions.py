import torch
from .utils_data import distort_images

class RandomNTIREDistortion:
    def __init__(self, max_distortions=3, num_levels=5, p=0.5):
        self.max_distortions = max_distortions
        self.num_levels = num_levels
        self.p = p  # probability to apply distortions

    def __call__(self, img: torch.Tensor):
        # img must be [3, H, W] in range [0,1]
        if torch.rand(1).item() < self.p:
            img, _, _ = distort_images(
                img,
                max_distortions=self.max_distortions,
                num_levels=self.num_levels
            )
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(max_distortions={self.max_distortions}, num_levels={self.num_levels}, p={self.p})"