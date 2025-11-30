from torch.utils.data import IterableDataset
from PIL import Image

import logging
logger = logging.getLogger(__name__)

class HuggingFaceIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, transform=None, platform=None, generator=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.platform = [p.lower() for p in platform] if platform else None
        self.generator = [g.lower().replace(" ", "").replace("-", "") for g in generator] if generator else None

    def __iter__(self):
        if isinstance(self.dataset[0]["image"], str):
            logger.debug("image is a string path")
        else:
            logger.error("image is something else:", type(item["image"]))
            raise ValueError(f"image is something else: {type(item["image"])}")

        for item in self.dataset:
            # Filter by platform
            if self.platform and item["platform"].lower() not in self.platform:
                continue

            # Filter by generator
            gen_name = item["generator"].lower().replace(" ", "").replace("-", "")
            if self.generator and gen_name not in self.generator:
                continue

            # Open image from path
            image = Image.open(item["image"]).convert("RGB")
            label = 0 if item["label"] == "real" else 1

            if self.transform:
                image = self.transform(image)

            yield image, label
