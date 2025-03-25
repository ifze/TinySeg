import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        
        # 创建金字塔池化分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        
        # 特征融合后的卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + out_channels*4, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size()[2:]
        
        # 处理所有分支并上采样
        branch_outs = []
        for branch in self.branches:
            out = branch(x)
            out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
            branch_outs.append(out)
        
        # 拼接特征
        out = torch.cat([x] + branch_outs, dim=1)
        return self.fusion(out)

class PSPNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 骨干网络（修改后的ResNet18）
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后两层
        
        # PSP模块
        self.psp = PSPModule(in_channels=512)
        
        # 最终预测层
        self.final = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.Upsample(size=128, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # 骨干网络提取特征
        features = self.backbone(x)  # [B, 512, 8, 8]
        
        # PSP模块处理
        psp_out = self.psp(features)  # [B, 512, 8, 8]
        
        # 最终上采样
        out = self.final(psp_out)    # [B, 5, 128, 128]
        return out

# 测试代码
if __name__ == "__main__":
    model = PSPNet()
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")  # 应该输出 [2, 5, 128, 128]