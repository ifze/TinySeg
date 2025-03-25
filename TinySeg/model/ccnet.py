import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # 添加连续化处理
        query = self._reshape_to_heads(query).contiguous()
        key = self._reshape_to_heads(key).contiguous()
        value = self._reshape_to_heads(value).contiguous()

        h_attn = self._compute_directional_attention(query, key, value, 'horizontal')
        v_attn = self._compute_directional_attention(query, key, value, 'vertical')

        attn = h_attn + v_attn
        attn = self._reshape_from_heads(attn, B)
        return self.gamma * attn + x

    def _reshape_to_heads(self, x):
        B, C, H, W = x.shape
        return x.reshape(B*self.num_heads, C//self.num_heads, H, W)  # 使用reshape

    def _reshape_from_heads(self, x, batch_size):
        B_H, C, H, W = x.shape
        return x.reshape(batch_size, C*self.num_heads, H, W)  # 使用reshape

    def _compute_directional_attention(self, query, key, value, direction):
        B_H, C, H, W = query.shape
        
        if direction == 'horizontal':
            query = query.permute(0, 2, 3, 1).contiguous()  # 添加连续化
            key = key.permute(0, 2, 1, 3).contiguous()
            energy = torch.matmul(query, key)
        else:
            query = query.permute(0, 3, 2, 1).contiguous()
            key = key.permute(0, 3, 1, 2).contiguous()
            energy = torch.matmul(query, key)

        attention = F.softmax(energy, dim=-1)
        value = value.permute(0, 2, 3, 1).contiguous() if direction == 'horizontal' else value.permute(0, 3, 2, 1).contiguous()
        out = torch.matmul(attention, value)
        return out.permute(0, 3, 1, 2).contiguous()  # 最后添加连续化

class CCNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 骨干网络（示例使用ResNet18）
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 输出尺寸 [B, 512, 8, 8]
        
        # 第一次Criss-Cross注意力
        self.cc_attention1 = CrissCrossAttention(512, num_heads=4)
        
        # 第二次Criss-Cross注意力
        self.cc_attention2 = CrissCrossAttention(512, num_heads=4)
        
        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(size=128, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # 骨干网络提取特征
        x = self.backbone(x)  # [B, 512, 8, 8]
        
        # 第一次注意力处理
        x = self.cc_attention1(x)
        
        # 第二次注意力处理
        x = self.cc_attention2(x)
        
        # 最终输出
        return self.final(x)

# 测试代码
if __name__ == "__main__":
    model = CCNet()
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    print(f"输入尺寸: {x.shape}")        # [2, 3, 128, 128]
    print(f"输出尺寸: {output.shape}")    # 应该输出 [2, 5, 128, 128]