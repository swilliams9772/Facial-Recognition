"""
Face embedding model built on a ResNet18 backbone.

The network replaces the final classification layer with a compact
embedding head that maps each face to a fixed-length vector in a
learned metric space. During recognition the embeddings are compared
with cosine similarity or Euclidean distance.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FaceEmbeddingNet(nn.Module):
    """ResNet18-based face embedding network.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the output embedding vector.
    pretrained : bool
        Whether to initialise the backbone with ImageNet weights.
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Everything except the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        # L2-normalise so cosine similarity == dot product
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
