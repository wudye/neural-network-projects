import torch
import torch.nn.functional as F


class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
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

        # ========== decoder (使用转置卷积) ==========
        # 转置卷积: stride=2 会使特征图尺寸翻倍
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1),  # 64(上采样) + 64(skip) = 128
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU()
        )

        self.upconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),  # 32(上采样) + 32(skip) = 64
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

        # ========== decoder (使用转置卷积上采样) ==========
        # 第一层上采样: 128 -> 64, 尺寸翻倍
        dec1 = self.upconv1(bottleneck)  # 使用转置卷积

        # 调整尺寸以匹配 skip connection (处理尺寸不匹配的情况)
        if dec1.shape[2:] != enc2.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc2.shape[2:], mode='bilinear', align_corners=False)

        # 拼接 skip connection
        dec1 = torch.cat([dec1, enc2], dim=1)  # 64 + 64 = 128
        dec1 = self.decoder1(dec1)

        # 第二层上采样: 64 -> 32, 尺寸翻倍
        dec2 = self.upconv2(dec1)

        if dec2.shape[2:] != enc1.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)

        dec2 = torch.cat([dec2, enc1], dim=1)  # 32 + 32 = 64
        dec2 = self.decoder2(dec2)

        # 最终输出
        output = F.interpolate(dec2, size=x.shape[2:], mode='bilinear', align_corners=False)
        output = self.output(output)

        return output
"""
import torch
import torch.nn.functional as F


class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
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

        # ========== decoder (使用转置卷积) ==========
        # 转置卷积 + 通道变换 合并在一起
        # ConvTranspose2d: stride=2 尺寸翻倍, 同时改变通道数
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1),  # 64(上采样) + 64(skip) = 128
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU()
        )

        self.upconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),  # 32(上采样) + 32(skip) = 64
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

        # ========== decoder (上采样 + 拼接 合并) ==========
        # 第一层: 转置卷积同时完成 尺寸翻倍(14->28) 和 通道变换(128->64)
        dec1 = self.upconv1(bottleneck)  # 128->64, 尺寸翻倍
        dec1 = torch.cat([dec1, enc2], dim=1)  # 64 + 64 = 128
        dec1 = self.decoder1(dec1)

        # 第二层: 转置卷积同时完成 尺寸翻倍(14->28) 和 通道变换(64->32)
        dec2 = self.upconv2(dec1)  # 64->32, 尺寸翻倍
        dec2 = torch.cat([dec2, enc1], dim=1)  # 32 + 32 = 64
        dec2 = self.decoder2(dec2)

        # 最终输出
        output = F.interpolate(dec2, size=x.shape[2:], mode='bilinear', align_corners=False)
        output = self.output(output)

        return output


"""