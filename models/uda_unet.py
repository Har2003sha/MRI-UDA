"""
UDA-UNet: A compact U-Net segmentation network combined with a
Gradient-Reversal-Layer (GRL) domain discriminator, implementing the
Domain-Adversarial Neural Network (DANN, Ganin et al. 2016) approach to
Unsupervised Domain Adaptation (UDA).

Training regime:
  - SOURCE domain (Scanner A): images + tumor masks -> segmentation loss
  - TARGET domain (Scanner B): images ONLY, no masks -> domain loss only
  - A domain classifier tries to tell source features from target features.
  - The Gradient Reversal Layer flips the gradient sign flowing back into
    the encoder, so the encoder is trained to make source and target
    features INDISTINGUISHABLE to the domain classifier -> the encoder
    learns scanner-invariant (domain-invariant) features, which is exactly
    what lets a model trained only on Scanner-A labels segment Scanner-B
    images well.
"""
import torch
import torch.nn as nn
from torch.autograd import Function


# --------------------------- Gradient Reversal ----------------------------
class GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


# ------------------------------- U-Net blocks ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------ Full model ---------------------------------
class UDAUNet(nn.Module):
    """
    Encoder -> bottleneck (shared, domain-invariant feature space)
             -> Decoder branch: segmentation mask
             -> GRL + Domain classifier branch: source(0) vs target(1)
    """

    def __init__(self, in_ch=1, base=12, num_classes=1):
        super().__init__()
        self.inc = ConvBlock(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.bottleneck = Down(base * 8, base * 16)

        self.up1 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8, base * 4)
        self.up3 = Up(base * 4, base * 2)
        self.up4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, num_classes, 1)

        # Domain classifier operates on globally-pooled bottleneck features
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 2 domains: source / target
        )

    def forward(self, x, alpha=0.0, return_domain=True):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        seg_logits = self.outc(d)

        domain_logits = None
        if return_domain:
            rev_feat = grad_reverse(x5, alpha)
            domain_logits = self.domain_classifier(rev_feat)

        return seg_logits, domain_logits


def dice_coefficient(pred_mask, gt_mask, eps=1e-6):
    pred_flat = pred_mask.reshape(-1).astype("float32")
    gt_flat = gt_mask.reshape(-1).astype("float32")
    inter = (pred_flat * gt_flat).sum()
    return float((2 * inter + eps) / (pred_flat.sum() + gt_flat.sum() + eps))


def iou_score(pred_mask, gt_mask, eps=1e-6):
    pred_flat = pred_mask.reshape(-1).astype("float32")
    gt_flat = gt_mask.reshape(-1).astype("float32")
    inter = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum() - inter
    return float((inter + eps) / (union + eps))
