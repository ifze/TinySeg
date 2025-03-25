import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # 定义四个并行分支
        self.branches = nn.ModuleList([
            # 分支1: 1x1卷积
            nn.Conv2d(in_channels, out_channels, 1),
            
            # 分支2: 3x3卷积 (dilation=6)
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            
            # 分支3: 3x3卷积 (dilation=12)
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            
            # 分支4: 全局池化分支
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                nn.Conv2d(in_channels, out_channels, 1),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
            )
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 保存原始尺寸用于上采样
        h, w = x.size()[2:]
        
        branch_outs = []
        for branch in self.branches:
            out = branch(x)
            # 对全局池化分支进行精确尺寸恢复
            if isinstance(branch, nn.Sequential):
                out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
            branch_outs.append(out)
        
        # 拼接所有分支输出
        out = torch.cat(branch_outs, dim=1)
        return self.fusion(out)

class DeepLabv3(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 骨干网络（使用ResNet18）
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 输出尺寸 [B, 512, 8, 8]
        
        # ASPP模块
        self.aspp = ASPPModule(in_channels=512, out_channels=256)
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # 骨干网络提取特征
        features = self.backbone(x)  # [B, 512, 8, 8]
        
        # ASPP处理
        aspp_out = self.aspp(features)  # [B, 256, 8, 8]
        
        # 最终上采样
        out = self.final(aspp_out)    # [B, 5, 128, 128]
        return out
