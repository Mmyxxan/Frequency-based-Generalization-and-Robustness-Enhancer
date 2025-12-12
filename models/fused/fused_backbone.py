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

        if self.fuse_technique in ["Concat", "Gated_concat"]:
            self._out_features = project_dim * len(self.backbone_list)
        elif self.fuse_technique == "Gated_fusion":
            self.gate_linear = nn.Linear(len(self.backbone_list) * project_dim, project_dim)
            self._out_features = project_dim
        elif self.fuse_technique == "Self_attention":
            self.self_attention = SelfAttention(embed_dim=project_dim)
            self._out_features = project_dim

    def forward(self, inputs, return_map=False):
        assert len(inputs) == len(self.backbones)
        outputs = []
        for i, input in enumerate(inputs):
            outputs.append(self.projections[i](self.backbones[i](input)))
        if self.fuse_technique == "Concat":
            return torch.cat(outputs, dim=1)
        elif self.fuse_technique == "Gated_fusion":
            # concat into (B, N*D)
            x = torch.cat(outputs, dim=1)
            # attention-like weights
            w = torch.softmax(self.gate_linear(x), dim=1)
            # stack original outputs → (B, N, D)
            O = torch.stack(outputs, dim=1)
            # fuse → (B, D)
            F = (w.unsqueeze(-1) * O).sum(dim=1)
            if return_map:
                return F, w
            return F
        elif self.fuse_technique == "Gated_concat":
            # concat into (B, N*D) for computing weights
            x = torch.cat(outputs, dim=1)
            # attention-like weights
            w = torch.softmax(self.gate_linear(x), dim=1)  # (B, N)
            # apply weights without summing
            weighted_outputs = [w[:, i].unsqueeze(-1) * outputs[i] for i in range(len(outputs))]
            # concatenate along feature dim → (B, N*D)
            F = torch.cat(weighted_outputs, dim=1)
            if return_map:
                return F, w
            return F     
        elif self.fuse_technique == "Self_attention":
            if return_map:
                return self.self_attention(outputs)
            else:
                return self.self_attention(outputs)[0]
        else:
            logger.error(f"Unknown fuse technique: {self.fuse_technique}")
            raise ValueError(f"Unknown fuse technique: {self.fuse_technique}")
