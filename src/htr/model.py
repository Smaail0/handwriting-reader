"""CRNN line recognizer: VGG-style CNN + sequence model + CTC head.

Built from ops that quantize well (Conv+BN+ReLU, MaxPool, Linear, LSTM or Conv1d).
Horizontal stride is 4, so a line of width W yields W // 4 output frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_classes: int
    height: int = 64
    channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    hidden: int = 256
    seq: str = "lstm"  # "lstm" (BiLSTM) or "conv" (dilated temporal convs, fully static-quantizable)
    seq_layers: int = 2
    dropout: float = 0.2
    arch: str = "vgg"  # "vgg" (original) or "lite" (stride-2 stem: ~4x fewer CNN multiply-adds)
    separable: bool = False  # lite only: depthwise-separable convs after the stem

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = dict(d)
        d["channels"] = tuple(d["channels"])
        return cls(**d)


def conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


def sep_conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    """Depthwise 3x3 + pointwise 1x1 (MobileNet-style)."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def build_backbone(cfg: "ModelConfig") -> nn.Sequential:
    """Both variants map (B, 1, H, W) -> (B, c4, H/16, W/4)."""
    c1, c2, c3, c4 = cfg.channels
    if cfg.arch == "vgg":
        return nn.Sequential(
            conv_bn_relu(1, c1), nn.MaxPool2d(2),                              # H/2,  W/2
            conv_bn_relu(c1, c2), nn.MaxPool2d(2),                             # H/4,  W/4
            conv_bn_relu(c2, c3), conv_bn_relu(c3, c3), nn.MaxPool2d((2, 1)),  # H/8
            conv_bn_relu(c3, c4), conv_bn_relu(c4, c4), nn.MaxPool2d((2, 1)),  # H/16
        )
    if cfg.arch == "lite":
        block = sep_conv_bn_relu if cfg.separable else conv_bn_relu
        return nn.Sequential(
            conv_bn_relu(1, c1, stride=2),                     # H/2,  W/2  (no full-resolution conv)
            block(c1, c2), nn.MaxPool2d(2),                    # H/4,  W/4
            block(c2, c3), block(c3, c3), nn.MaxPool2d((2, 1)),  # H/8
            block(c3, c4), block(c4, c4), nn.MaxPool2d((2, 1)),  # H/16
        )
    raise ValueError(f"unknown arch {cfg.arch!r}")


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel // 2), dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return x + self.drop(self.act(self.bn(self.conv(x))))


class CRNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.height % 16:
            raise ValueError("height must be a multiple of 16")
        self.cfg = cfg
        self.cnn = build_backbone(cfg)
        self.proj = nn.Linear(cfg.channels[3] * (cfg.height // 16), cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.seq == "lstm":
            self.seq = nn.LSTM(cfg.hidden, cfg.hidden, num_layers=cfg.seq_layers, bidirectional=True,
                               batch_first=True, dropout=cfg.dropout if cfg.seq_layers > 1 else 0.0)
            out_dim = 2 * cfg.hidden
        elif cfg.seq == "conv":
            dilations = [1, 2, 4] * cfg.seq_layers
            self.seq = nn.Sequential(*[TemporalConvBlock(cfg.hidden, 5, d, cfg.dropout) for d in dilations])
            out_dim = cfg.hidden
        else:
            raise ValueError(f"unknown seq type {cfg.seq!r}")
        self.head = nn.Linear(out_dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> logits (B, W // 4, num_classes)."""
        f = self.cnn(x)                          # (B, C, h, T)
        f = f.permute(0, 3, 1, 2).flatten(2)     # (B, T, C*h)
        f = self.drop(torch.relu(self.proj(f)))  # (B, T, hidden)
        if self.cfg.seq == "lstm":
            f, _ = self.seq(f)
        else:
            f = self.seq(f.transpose(1, 2)).transpose(1, 2)
        return self.head(self.drop(f))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, map_location="cpu"):
    from .charset import Charset

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = CRNN(ModelConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Charset(ckpt["charset"]), ckpt
