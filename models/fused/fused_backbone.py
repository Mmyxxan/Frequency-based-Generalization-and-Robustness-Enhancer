from .backbone import Backbone

import torch.nn as nn
import torch

from utils import logger

class FusedBackbone(Backbone):

    def preprocess(self, shared_tensor):
        return [backbone_cls.preprocess(shared_tensor) for backbone_cls in self.backbone_list]

    def __init__(self, backbone_list, project_dim, fuse_technique, freeze=True, pretrained=True, **kwargs):
        super().__init__()
        self.backbone_list = backbone_list
        self.fuse_technique = fuse_technique

        self.backbones = nn.ModuleList()
        self.projections = nn.ModuleList()

        for backbone_cls in self.backbone_list:
            model = backbone_cls(freeze=freeze, pretrained=pretrained)
            projection = nn.Linear(model._out_features, project_dim)
            self.backbones.append(model)
            self.projections.append(projection)

        self._out_features = project_dim * len(self.backbone_list)

    def forward(self, inputs):
        assert len(inputs) == len(self.backbones)
        outputs = []
        for i, input in enumerate(inputs):
            outputs.append(self.projections[i](self.backbones[i](input)))
        if self.fuse_technique == "Concat":
            return torch.cat(outputs, dim=1)
        else:
            logger.error(f"Unknown fuse technique: {self.fuse_technique}")
            raise ValueError(f"Unknown fuse technique: {self.fuse_technique}")
