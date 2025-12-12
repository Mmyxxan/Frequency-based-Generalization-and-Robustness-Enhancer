from .backbone import Backbone

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import logger

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear layers to project input to Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, X):
        """
        X: (batch_size, seq_len, embed_dim)
        """
        Q = self.W_q(X)  # (B, N, D)
        K = self.W_k(X)  # (B, N, D)
        V = self.W_v(X)  # (B, N, D)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # (B, N, N)
        weights = F.softmax(scores, dim=-1)  # attention weights (B, N, N)

        # Weighted sum of values
        output = torch.matmul(weights, V)  # (B, N, D)
        return output, weights

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

        num_backbones = len(self.backbone_list)

        if self.fuse_technique == "Concat":
            self._out_features = project_dim * num_backbones

        elif self.fuse_technique in ["Gated_fusion", "Gated_concat"]:
            # Support only exactly 2 backbones for gated fusion/concat
            if num_backbones != 2:
                logger.error(f"{self.fuse_technique} requires exactly 2 backbones, but got {num_backbones}")
                raise ValueError(f"{self.fuse_technique} requires exactly 2 backbones, but got {num_backbones}")
            # Linear layer projects concatenated features (2*D) to D
            self.gate_linear = nn.Linear(2 * project_dim, project_dim)

            if self.fuse_technique == "Gated_fusion":
                self._out_features = project_dim
            else:  # Gated_concat
                self._out_features = 2 * project_dim

        elif self.fuse_technique == "Self_attention":
            self.self_attention = SelfAttention(embed_dim=project_dim)
            self._out_features = project_dim

        else:
            logger.error(f"Unknown fuse technique: {self.fuse_technique}")
            raise ValueError(f"Unknown fuse technique: {self.fuse_technique}")

    def forward(self, inputs, return_map=False):
        assert len(inputs) == len(self.backbones)
        outputs = []
        for i, input in enumerate(inputs):
            outputs.append(self.projections[i](self.backbones[i](input)))

        if self.fuse_technique == "Concat":
            # Simple concatenation along feature dimension
            return torch.cat(outputs, dim=1)

        elif self.fuse_technique == "Gated_fusion":
            # Exactly 2 backbones only
            assert len(outputs) == 2
            x = torch.cat(outputs, dim=1)  # (B, 2*D)
            g = torch.sigmoid(self.gate_linear(x))  # (B, D), gate weights between 0 and 1
            A, B = outputs  # each (B, D)
            F = g * A + (1 - g) * B  # gated fusion per feature dim
            if return_map:
                return F, g
            return F

        elif self.fuse_technique == "Gated_concat":
            # Exactly 2 backbones only
            assert len(outputs) == 2
            x = torch.cat(outputs, dim=1)  # (B, 2*D)
            g = torch.sigmoid(self.gate_linear(x))  # (B, D)
            weighted_outputs = [g * outputs[0], (1 - g) * outputs[1]]  # weighted but not summed
            F = torch.cat(weighted_outputs, dim=1)  # (B, 2*D)
            if return_map:
                return F, g
            return F

        elif self.fuse_technique == "Self_attention":
            # inputs to self_attention must be (B, N, D)
            stacked = torch.stack(outputs, dim=1)
            if return_map:
                return self.self_attention(stacked)
            else:
                return self.self_attention(stacked)[0]

        else:
            logger.error(f"Unknown fuse technique: {self.fuse_technique}")
            raise ValueError(f"Unknown fuse technique: {self.fuse_technique}")
