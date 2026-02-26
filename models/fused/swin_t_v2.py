from .backbone import Backbone
from torchvision import transforms
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

class SwinTV2(Backbone):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def preprocess(image_tensor):
        return SwinTV2.normalize(image_tensor.clone())

    def __init__(self, freeze=True, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        else:
            self.model = swin_v2_t(weights=None)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._out_features = self.model.head.in_features

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        x = self.model.flatten(x)
        return x
