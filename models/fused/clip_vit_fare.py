import open_clip
from torchvision import transforms
from .backbone import Backbone

# import logging
# logger = logging.getLogger(__name__)

class CLIPViT_FARE(Backbone):

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    @staticmethod
    def preprocess(image_tensor):
        return CLIPViT_FARE.normalize(image_tensor.clone())

    def __init__(self, model_name="hf-hub:chs20/fare4-clip", pretrained=True, freeze=True):
        super().__init__()

        if pretrained:
            self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        else:
            self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=None)
        self.visual = self.model.visual

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._out_features = self.visual.ln_post.normalized_shape[0]
        # logger.debug(self.out_features) # 1024

    def forward(self, x):
        x = self.visual._embeds(x)
        x = self.visual.transformer(x)
        pooled, _ = self.visual._pool(x)
        return pooled
