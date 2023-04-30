import torch
from torch import nn
from transformers import DetrConfig, DetrForObjectDetection

class Detr(nn.Module):
    def __init__(self):
        super(Detr, self).__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def forward(self, x):
        out = self.model(x)
        return out