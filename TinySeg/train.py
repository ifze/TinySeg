import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from dataloader import create_dataloaders
from model.pspnet import PSPNet
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class Trainer:
    def __init__(self, model, optimizer, criterion, device, num_classes, classes):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.classes = classes
        self.train_loss_history = []
        self.val_miou_history = []
        self.best_model = None
        self.all_preds = []
        self.all_labels = []
        
    def _calculate_miou(self, outputs, labels):
        preds = outputs.argmax(1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        # 收集所有预测和标签用于混淆矩阵
        self.all_preds.extend(preds.flatten())
        self.all_labels.extend(labels.flatten())
        
        # 计算mIoU
        cm = confusion_matrix(labels.flatten(), preds.flatten(), labels=np.arange(self.num_classes))
        with np.errstate(invalid='ignore'):
            iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-10)
        return np.nanmean(iou)
    
    def _update_plots(self):
        plt.figure(figsize=(18, 6))
        
        # 训练曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss_history, 'b-', label='Train Loss')
        plt.plot(self.val_miou_history, 'r-', label='Val mIoU')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # 混淆矩阵
        plt.subplot(1, 3, 2)
        cm = confusion_matrix(self.all_labels, self.all_preds, normalize='true')
        sns.heatmap(cm, annot=True, fmt=".2%", cmap='Blues', 
                   xticklabels=self.classes, 
                   yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # 示例预测可视化
        plt.subplot(1, 3, 3)
        sample_image = self.last_batch[0].cpu().permute(1, 2, 0).numpy()
        sample_image = (sample_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # 反归一化
        sample_pred = self.last_pred[0].argmax(0).cpu().numpy()
        sample_gt = self.last_gt[0].cpu().numpy()
        
        plt.imshow(sample_image)
        plt.imshow(sample_pred, alpha=0.3, cmap='jet')
        plt.title('Sample Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # 保留最后一个batch用于可视化
            self.last_batch = images[:1].detach().cpu()
            self.last_pred = outputs[:1].detach().cpu()
            self.last_gt = labels[:1].detach().cpu()
            
        return epoch_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_miou = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                total_miou += self._calculate_miou(outputs, labels)
        return total_miou / len(val_loader)

    def run(self, train_loader, val_loader, epochs):
        self.all_preds = []
        self.all_labels = []
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)
            
            # 验证阶段
            val_miou = self.validate(val_loader)
            self.val_miou_history.append(val_miou)
            
            # 更新最佳模型
            if val_miou >= max(self.val_miou_history, default=0):
                self.best_model = self.model.state_dict()
                torch.save(self.best_model, f'best_model_epoch{epoch+1}.pth')
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val mIoU: {val_miou:.4f}")
            
            # 每3个epoch更新可视化
            if (epoch+1) % 25 == 0:
                self._update_plots()
        
        # 最终可视化
        self._update_plots()
        torch.save(self.best_model, 'final_best_model.pth')

# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'data_root': 'E:/code/TinySeg/TinySeg_data',  
        'batch_size': 16,
        'num_workers': 4,
        'num_classes': 6,
        'classes': ['背景', '飞机', '猫', '人类', '车辆','鸟类'],  # 根据实际类别修改
        'epochs': 50,
        'lr': 0.001,
        'augment_mode': 'full'
    }

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment_mode=config['augment_mode']
    )
    
    # 初始化模型
    model = PSPNet(num_classes=config['num_classes']).to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0, 3.0, 1.0,1.0]).to(device))  # 根据类别不平衡调整
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=config['num_classes'],
        classes=config['classes']
    )
    
    # 运行训练
    trainer.run(train_loader, val_loader, config['epochs'])