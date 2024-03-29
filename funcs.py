from torch import nn
from attn_blocks import CBAM

FUNCS = {
    "conv": nn.Conv2d,
    "bn": nn.BatchNorm2d,
    "relu": nn.ReLU,
    "cbam": CBAM,
    "max_pool": nn.MaxPool2d,
    "avg_pool": nn.AvgPool2d,
    "dropout": nn.Dropout2d,
    "fc": nn.Linear,
}
