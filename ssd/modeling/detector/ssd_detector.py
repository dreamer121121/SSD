from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)#backbone提取特征
        self.box_head = build_box_head(cfg)#head

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
