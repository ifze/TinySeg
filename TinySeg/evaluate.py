import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

class SegmentationEvaluator:
    def __init__(self, num_classes, class_names, colors=None):
        self.num_classes = num_classes
        self.class_names = class_names
        self.colors = colors or plt.cm.tab20.colors[:num_classes]
        self.confusion = np.zeros((num_classes, num_classes))
        self.sample_cache = []  # 存储可视化样本

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes))
        self.sample_cache.clear()

    def update(self, images, preds, trues):
        """ 更新评估指标并缓存样本 """
        # 更新混淆矩阵
        pred_flat = preds.flatten().cpu().numpy()
        true_flat = trues.flatten().cpu().numpy()
        self.confusion += confusion_matrix(true_flat, pred_flat, 
                                         labels=np.arange(self.num_classes))
        
        # 缓存可视化样本（保留前3个样本）
        if len(self.sample_cache) < 3:
            for img, pred, true in zip(images, preds, trues):
                if len(self.sample_cache) >= 3:
                    break
                self.sample_cache.append((
                    img.cpu().permute(1,2,0).numpy(),
                    pred.cpu().numpy(),
                    true.cpu().numpy()
                ))

    def pixel_accuracy(self):
        """ Pixel Accuracy = Σ对角线元素 / Σ所有元素 """
        correct = np.diag(self.confusion).sum()
        total = self.confusion.sum()
        return correct / (total + 1e-6)

    def mean_iou(self):
        """ mIoU = (1/C) * Σ[TP_c / (TP_c + FP_c + FN_c)] """
        ious = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp
            iou = tp / (tp + fp + fn + 1e-6)
            ious.append(iou)
        return np.mean(ious)

    def dice_coefficient(self):
        """ Dice = (2*TP_c) / (2*TP_c + FP_c + FN_c) 的平均值 """
        dices = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            dices.append(dice)
        return np.mean(dices)

    def visualize_results(self, save_path="evaluation_results.png"):
        """ 生成符合图示要求的可视化结果 """
        plt.figure(figsize=(18, 6))
        ax = plt.gca()
        
        # 1. 可视化样本
        for i, (img, pred, true) in enumerate(self.sample_cache):
            # 图像反归一化
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            
            # 创建子图
            plt.subplot(3, 4, i*4+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Sample {i+1} Input")
            
            plt.subplot(3, 4, i*4+2)
            plt.imshow(true, cmap=ListedColormap(self.colors), vmin=0, vmax=self.num_classes-1)
            plt.axis('off')
            plt.title("Ground Truth")
            
            plt.subplot(3, 4, i*4+3)
            plt.imshow(pred, cmap=ListedColormap(self.colors), vmin=0, vmax=self.num_classes-1)
            plt.axis('off')
            plt.title("Prediction")

        # 2. 评估指标公式和数值
        metrics_text = (
            "Evaluation Metrics:\n\n"
            "1. Pixel Accuracy:\n"
            r"$\frac{\sum_{c} TP_c}{\sum_{c}(TP_c + FP_c + FN_c)} = " 
            f"{self.pixel_accuracy():.3f}$\n\n"
            "2. mIoU:\n"
            r"$\frac{1}{C}\sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c + FN_c} = "
            f"{self.mean_iou():.3f}$\n\n"
            "3. Dice Coefficient:\n"
            r"$\frac{1}{C}\sum_{c=1}^{C} \frac{2TP_c}{2TP_c + FP_c + FN_c} = "
            f"{self.dice_coefficient():.3f}$"
        )
        
        plt.subplot(1, 4, 4)
        plt.text(0.1, 0.5, metrics_text, fontsize=12, va='center', linespacing=1.8)
        plt.axis('off')
        
        # 3. 颜色图例
        plt.subplot(3, 4, 12)
        for c, color in enumerate(self.colors):
            plt.plot([], [], color=color, label=f"{c}-{self.class_names[c]}")
        plt.legend(loc='center', frameon=False)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# 使用示例
if __name__ == "__main__":
    # 假设已有模型和数据
    CLASS_NAMES = ['背景', '道路', '建筑', '植被']
    COLORS = ['#000000', '#FF0000', '#00FF00', '#0000FF']
    
    evaluator = SegmentationEvaluator(
        num_classes=4,
        class_names=CLASS_NAMES,
        colors=COLORS
    )
    
    # 模拟评估过程
    for images, masks in val_loader:
        preds = model(images)
        preds = torch.argmax(preds, dim=1)
        evaluator.update(images, preds, masks)
    
    # 生成可视化报告
    print(f"Pixel Accuracy: {evaluator.pixel_accuracy():.4f}")
    print(f"mIoU: {evaluator.mean_iou():.4f}")
    print(f"Dice: {evaluator.dice_coefficient():.4f}")
    evaluator.visualize_results()