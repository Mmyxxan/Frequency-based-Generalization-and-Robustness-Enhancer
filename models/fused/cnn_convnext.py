from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision import transforms

from .backbone import Backbone

class ConvNeXtSmall(Backbone):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def preprocess(image_tensor):
        return ConvNeXtSmall.normalize(image_tensor.clone())  # Clone to avoid in-place ops

    def __init__(self, freeze=True, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        else:
            self.model = convnext_small(weights=None)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.norm = self.model.classifier[0]
        self.flatten = self.model.classifier[1]
        self.linear = self.model.classifier[2]
        self._out_features = self.linear.in_features

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)
        return x
