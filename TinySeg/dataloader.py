import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip,
    Rotate,  # 新版本名称
    RandomBrightnessContrast
)
LABEL_MAPPING = {0:0, 14:1, 38:2, 52:3, 75:4, 89:5}
LUT = np.zeros(256, dtype=np.int64)
for orig, new in LABEL_MAPPING.items():
    LUT[orig] = new

class TinySegDataset(Dataset):
    def __init__(self, jpeg_root, anno_root, split_file, transform=None, target_size=(128, 128)):
        """
        Args:
            jpeg_root (str): JPEGImages目录路径
            anno_root (str): Annotations目录路径
            split_file (str): ImageSets中的划分文件路径（如train.txt）
            transform (albumentations.Compose): 数据增强组合
        """
        self.target_size = target_size  
        self.jpeg_root = jpeg_root
        self.anno_root = anno_root
        self.transform = transform
        
        # 从ImageSets文件读取样本列表
        with open(split_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
        # 验证数据完整性
        self._validate_data_files()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_name = self.samples[idx]
        img_path = os.path.join(self.jpeg_root, f"{base_name}.jpg")
        mask_path = os.path.join(self.anno_root, f"{base_name}.png")

        # 读取数据并检查文件有效性
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"图像文件无法读取: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE),
        mask = LUT[mask] 
        if mask is None:
            raise FileNotFoundError(f"掩码文件无法读取: {mask_path}")

        # 调整尺寸到统一分辨率
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # 此时是Tensor
            mask = transformed['mask']    # 此时是Tensor
            mask = mask.long()  
        else:
            # 手动转换为Tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    def _validate_data_files(self):
        """验证所有数据文件是否存在"""
        missing = []
        for base_name in self.samples:
            img_path = os.path.join(self.jpeg_root, f"{base_name}.jpg")
            mask_path = os.path.join(self.anno_root, f"{base_name}.png")
            
            if not os.path.exists(img_path):
                missing.append(img_path)
            if not os.path.exists(mask_path):
                missing.append(mask_path)
        
        if missing:
            raise FileNotFoundError(f"缺失 {len(missing)} 个数据文件，例如：{missing[:3]}")

def create_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4,
    augment_mode='full'
):
    """创建适配新结构的数据加载器
    
    Args:
        data_root (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        augment_mode (str): 数据增强模式 ['none', 'base', 'full']
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 定义路径常量
    JPEG_ROOT = os.path.join(data_root, 'JPEGImages')
    ANNO_ROOT = os.path.join(data_root, 'Annotations')
    IMAGESETS_ROOT = os.path.join(data_root, 'ImageSets')
    
    # 创建数据集
    train_ds = TinySegDataset(
        jpeg_root=JPEG_ROOT,
        anno_root=ANNO_ROOT,
        split_file=os.path.join(IMAGESETS_ROOT, 'train.txt'),
        transform=get_transforms(augment_mode),
        target_size=(128, 128)  # 明确指定尺寸
    )
    
    val_ds = TinySegDataset(
        jpeg_root=JPEG_ROOT,
        anno_root=ANNO_ROOT,
        split_file=os.path.join(IMAGESETS_ROOT, 'val.txt'),
        transform=get_transforms('none'),
        target_size=(128, 128)  # 明确指定尺寸
    )
    
    # 创建数据加载器（保持原有配置）
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_transforms(config='none'):
    """获取数据增强组合
    
    Args:
        config (str): 增强配置 ['none', 'base', 'full']
    
    Returns:
        albumentations.Compose: 数据增强组合
    """
    target_size = (128, 128)
    common_transforms = [
        A.Resize(height=target_size[0], width=target_size[1], 
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if config == 'none':
        return A.Compose(common_transforms)
    
    base_aug = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.3
        )
    ]
    
    if config == 'base':
        return A.Compose(base_aug + common_transforms)
    
    if config == 'full':
        advanced_aug = [
            A.ElasticTransform(
                alpha=50, 
                sigma=50, 
                p=0.3,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.GridDistortion(
                num_steps=5, 
                distort_limit=0.3, 
                p=0.3,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.Compose([
                A.RandomCrop(width=96, height=96, p=0.5),
                A.Resize(
                    width=128, 
                    height=128,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST
                )
            ], p=0.5)
        ]
        return A.Compose(base_aug + advanced_aug + common_transforms)
