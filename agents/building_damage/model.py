import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # in_channels after upsampling + fused skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseDamageUNet(nn.Module):
    """
    Siamese U-Net architecture for pre- and post-disaster building damage segmentation.
    Uses a shared encoder (e.g. ResNet34) for pre and post imagery, fuses multi-scale
    features via concatenation & absolute difference, and decodes to predict per-pixel damage classes.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 5,
    ):
        super().__init__()
        
        # Shared feature extractor from smp
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        
        # ResNet encoder channel stages: e.g. [3, 64, 64, 128, 256, 512]
        enc_channels = self.encoder.out_channels
        
        # Fusion reduces [pre, post, |pre - post|] (3 * C) to C via 1x1 conv
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 3, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
            for c in enc_channels
        ])
        
        # Decoder stages
        # enc_channels: e0 (3), e1 (64), e2 (64), e3 (128), e4 (256), e5 (512)
        self.dec5 = DecoderBlock(in_channels=enc_channels[5], skip_channels=enc_channels[4], out_channels=256)
        self.dec4 = DecoderBlock(in_channels=256, skip_channels=enc_channels[3], out_channels=128)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=enc_channels[2], out_channels=64)
        self.dec2 = DecoderBlock(in_channels=64, skip_channels=enc_channels[1], out_channels=32)
        self.dec1 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)
        
        # Segmentation head
        self.head = nn.Conv2d(16, classes, kernel_size=1)

    def _fuse_features(self, feat_pre, feat_post, fusion_layer):
        diff = torch.abs(feat_pre - feat_post)
        combined = torch.cat([feat_pre, feat_post, diff], dim=1)
        return fusion_layer(combined)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        # Extract multi-scale feature pyramids
        pre_feats = self.encoder(pre)
        post_feats = self.encoder(post)
        
        # Fuse corresponding level features
        fused = [
            self._fuse_features(p, po, fusion)
            for p, po, fusion in zip(pre_feats, post_feats, self.fusions)
        ]
        # fused has stages [f0, f1, f2, f3, f4, f5]
        
        x = self.dec5(fused[5], fused[4])
        x = self.dec4(x, fused[3])
        x = self.dec3(x, fused[2])
        x = self.dec2(x, fused[1])
        x = self.dec1(x)
        
        logits = self.head(x)
        return logits


def build_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=5):
    """Factory function for building the Siamese damage segmentation model."""
    return SiameseDamageUNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


if __name__ == "__main__":
    model = build_model(encoder_weights=None)
    print("Building Damage Siamese model created successfully.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Smoke test forward pass
    dummy_pre = torch.randn(2, 3, 512, 512)
    dummy_post = torch.randn(2, 3, 512, 512)
    out = model(dummy_pre, dummy_post)
    print(f"Input shape: {dummy_pre.shape}, Output shape: {out.shape}")
    assert out.shape == (2, 5, 512, 512), f"Unexpected shape {out.shape}"
    print("Forward pass verification passed!")
