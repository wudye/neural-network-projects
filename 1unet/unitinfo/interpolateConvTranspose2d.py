import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 创建一个简单的测试图
x = torch.zeros(1, 1, 4, 4)
x[0, 0, 1:3, 1:3] = 1  # 中间一个小方块

# interpolate 上采样
y1 = F.interpolate(x, scale_factor=2, mode='nearest')
print("Interpolate (nearest):")
print(y1[0, 0])  # 邻近插值

# ConvTranspose2d 上采样
conv_t = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
y2 = conv_t(x.float())
print("\nConvTranspose2d:")
print(y2[0, 0].detach())
