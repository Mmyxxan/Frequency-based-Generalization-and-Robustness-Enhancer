import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

from .backbone import Backbone
from utils import logger

class ResNet50(Backbone):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def preprocess(image_tensor):
        return ResNet50.normalize(image_tensor.clone())  # Clone to avoid in-place ops

    def __init__(self, freeze=True, pretrained=True, resnet50_am_weights=None, map_location=None):
        super().__init__()
        if pretrained:
            if resnet50_am_weights:
                # Path to AM model from Google Research
                logger.info(f"Loading AugMix pretrained weights of ResNet50 on ImageNet from {resnet50_am_weights}...")
                checkpoint = torch.load(resnet50_am_weights, map_location=map_location, weights_only=False)
                # epoch = checkpoint["epoch"]
                # model = checkpoint["model"]
                state_dict = checkpoint["state_dict"]
                # best_acc1 = checkpoint["best_acc1"]
                # optimizer = checkpoint["optimizer"]
                # Remove 'module.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[len("module."):]
                    new_state_dict[k] = v
                self.model = resnet50(weights=None)
                self.model.load_state_dict(new_state_dict)
            else:
                # Default weights from PyTorch
                self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet50()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._out_features = 2048

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
