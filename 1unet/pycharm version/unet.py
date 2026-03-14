import torch
import torch.nn.functional as F


class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels accepts only 1 channel for color is only gray, not rgb
        # out_channels need 32 kernels(filters) output for (1, height, weight) to (32, height, weight)
        # kernel is 3 * 3
        # Padding refers to adding extra "fake" pixels (usually zeros) around the border of your input image before the kernel slides over it.

        # encoder
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
#normalizes the inputs of each layer to have a mean of 0 and standard deviation of 1 (during training).
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU()
        )
        self.pool1 = torch.nn.MaxPool2d(2)

        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
        )
        # seperate pooling layer, not in sequential, because we need to
        # save the output of encoder2 for skip connection
        self.pool2 = torch.nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU()
        )

        # ========== decoder ==========
        # key：in decoder1 ，after interpolate need change channels by conv
        self.decoder1 = torch.nn.Sequential(
            # input: 128 channels (64 interpolate + 64 skip)
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU()
        )

        self.decoder2 = torch.nn.Sequential(
            # input: 64 channels (32 interpolate + 32 skip)
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU()
        )

        self.output = torch.nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # encode
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        # ========== decoder  ==========
        # : interpolate upsampling
        dec1 = F.interpolate(bottleneck, size=enc2.shape[2:], mode='bilinear', align_corners=False)

        #  use 1x1 conv 128  → 64
        dec1 = self._upconv_channels(dec1, 64)  # 128 → 64

        # concatenate skip connection
        dec1 = torch.cat([dec1, enc2], dim=1)  # 64 + 64 = 128
        dec1 = self.decoder1(dec1)

        # second decoder
        dec2 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self._upconv_channels(dec2, 32)  # 64 → 32
        dec2 = torch.cat([dec2, enc1], dim=1)  # 32 + 32 = 64
        dec2 = self.decoder2(dec2)

        # end output
        output = F.interpolate(dec2, size=x.shape[2:], mode='bilinear', align_corners=False)
        output = self.output(output)

        return output

    def _upconv_channels(self, x, out_channels):
        """用 1x1 卷积改变通道数"""
        if x.shape[1] != out_channels:
            x = torch.nn.Conv2d(x.shape[1], out_channels, 1)(x)
        return x


if __name__ == "__main__":
    unet = Unet()
    image_tensor = torch.randn(5, 1, 28, 28)
    output = unet(image_tensor)
    print(f"input: {image_tensor.shape}")
    print(f"output: {output.shape}")
