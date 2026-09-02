import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ──────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample: MaxPool2d (halves spatial size) → DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upsample (doubles spatial size) → concatenate skip connection → DoubleConv.
    Uses bilinear upsampling (lighter than transposed convolution).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # handle odd input sizes — pad if needed before concatenating
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])

        x = torch.cat([skip, x], dim=1)   # concat along channel axis
        return self.conv(x)


# ── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Vanilla U-Net for multiclass segmentation.

    Architecture:
        Encoder:  1 → 64 → 128 → 256 → 512
        Bottleneck:        512 → 1024
        Decoder:  1024 → 512 → 256 → 128 → 64
        Head:     64  → num_classes  (1x1 conv, no activation)

    Input  : (B, 1,           256, 256)
    Output : (B, num_classes, 256, 256)  — raw logits for CrossEntropyLoss
    """

    def __init__(self, in_channels=1, num_classes=3, features=[64, 128, 256, 512]):
        super().__init__()

        # ── encoder ──
        self.inc   = DoubleConv(in_channels, features[0])   # 1   → 64
        self.down1 = Down(features[0], features[1])          # 64  → 128
        self.down2 = Down(features[1], features[2])          # 128 → 256
        self.down3 = Down(features[2], features[3])          # 256 → 512

        # ── bottleneck ──
        self.down4 = Down(features[3], features[3] * 2)     # 512 → 1024

        # ── decoder ──
        self.up1   = Up(features[3] * 2 + features[3], features[3])   # 1024+512 → 512
        self.up2   = Up(features[3] + features[2],     features[2])   # 512+256  → 256
        self.up3   = Up(features[2] + features[1],     features[1])   # 256+128  → 128
        self.up4   = Up(features[1] + features[0],     features[0])   # 128+64   → 64

        # ── output head ──
        self.outc  = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # encoder — save skip connections
        x1 = self.inc(x)      # (B, 64,   256, 256)
        x2 = self.down1(x1)   # (B, 128,  128, 128)
        x3 = self.down2(x2)   # (B, 256,   64,  64)
        x4 = self.down3(x3)   # (B, 512,   32,  32)
        x5 = self.down4(x4)   # (B, 1024,  16,  16)

        # decoder — pass skip connections
        x = self.up1(x5, x4)  # (B, 512,  32,  32)
        x = self.up2(x,  x3)  # (B, 256,  64,  64)
        x = self.up3(x,  x2)  # (B, 128, 128, 128)
        x = self.up4(x,  x1)  # (B, 64,  256, 256)

        return self.outc(x)    # (B, 3,   256, 256)  raw logits


# ── sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {total_params:,}")

    dummy = torch.randn(2, 1, 256, 256)   # batch of 2
    out   = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")   # should be (2, 3, 256, 256)