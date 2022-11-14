from lit_models.build import build_model
from models.lknet import *
import torch.nn as nn
import torch


class LKCT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.LK = create_LKNet(drop_path_rate=0.1, num_classes=1, use_checkpoint=False,
                                   small_kernel_merged=False)
        self.LIT = build_model('lit')
        self.head = nn.Linear(2, self.num_classes)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        out1 = self.LK(x)
        out1 = self.dp(out1)
        out2 = self.LIT(x)
        out2 = self.dp(out2)
        out = torch.cat([out1, out2], dim=1)
        out = self.head(out)
        return out
