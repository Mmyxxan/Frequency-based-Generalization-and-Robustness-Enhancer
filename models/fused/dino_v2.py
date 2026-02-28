from .backbone import Backbone
import torch
from torchvision import transforms

class DINOV2(Backbone):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def preprocess(image_tensor):
        return DINOV2.normalize(image_tensor.clone())

    def __init__(self, freeze=True, pretrained=True, model_name="dinov2_vitb14"):
        super().__init__()

        if pretrained:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name
            )
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=False
            )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Output dimension depends on model
        if "vits" in model_name:
            self._out_features = 384
        elif "vitb" in model_name:
            self._out_features = 768
        elif "vitl" in model_name:
            self._out_features = 1024
        elif "vitg" in model_name:
            self._out_features = 1536

    def forward(self, x):
        """
        Returns CLS token embedding (global image feature)
        """
        features = self.model(x)   # (B, N+1, D)
        cls_token = features[:, 0] # (B, D)
        return cls_token
        