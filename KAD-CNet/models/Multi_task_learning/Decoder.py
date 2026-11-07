"""Auxiliary decoders of Multi-task Guided Learning (MGL).

Both decoders take five backbone features (fine to coarse) and are used only during training.
"""
import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """Upsample by 2, concatenate the skip feature, then two 3x3 conv-BN-ReLU layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = torch.cat([self.upsample(x), skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(x)))


class UNetDecoder_IA(nn.Module):
    """Instance Awareness (IA) branch: predicts a foreground mask at input resolution."""

    def __init__(self, encoder_channels, out_channels=1):
        super().__init__()
        decoder_channels = encoder_channels[::-1]
        self.decoder_channels = decoder_channels
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(decoder_channels[i], decoder_channels[i + 1]) for i in range(len(decoder_channels) - 1))
        self.final_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1))
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, encoded_features):
        features = encoded_features[::-1]
        x = features[0]
        for block, skip in zip(self.decoder_blocks, features[1:]):
            x = block(x, skip)
        return self.final_conv(self.final_decoder(x))  # logits


class UNetDecoder_DP(nn.Module):
    """Detail Preservation (DP) branch: reconstructs the image at twice the input resolution."""

    def __init__(self, encoder_channels, out_channels=1):
        super().__init__()
        decoder_channels = encoder_channels[::-1]
        self.decoder_channels = decoder_channels
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(decoder_channels[i], decoder_channels[i + 1]) for i in range(len(decoder_channels) - 1))
        self.final_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1))
        self.sr = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 4 * decoder_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2))
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, encoded_features):
        features = encoded_features[::-1]
        x = features[0]
        for block, skip in zip(self.decoder_blocks, features[1:]):
            x = block(x, skip)
        return self.final_conv(self.sr(self.final_decoder(x)))
