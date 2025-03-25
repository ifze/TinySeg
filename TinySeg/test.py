import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def analyze_labels(anno_dir):
    """
    分析标签分布
    :param anno_dir: 标签文件目录路径
    :return: 标签统计字典，异常文件列表
    """
    # 初始化统计器
    label_counter = defaultdict(int)
    bad_files = []
    
    # 获取所有标签文件
    all_files = [f for f in os.listdir(anno_dir) if f.endswith('.png')]
    print(f"发现 {len(all_files)} 个标签文件")

    # 遍历所有标签文件
    for filename in tqdm(all_files, desc="分析标签"):
        filepath = os.path.join(anno_dir, filename)
        
        try:
            # 以灰度模式读取文件
            label = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise IOError(f"无法读取文件: {filename}")
                
            # 检查是否为单通道
            if len(label.shape) != 2:
                raise ValueError(f"非灰度图像: {filename} (形状: {label.shape})")
            
            # 统计像素值
            unique, counts = np.unique(label, return_counts=True)
            for val, cnt in zip(unique, counts):
                label_counter[val] += cnt
                
            # 记录异常文件
            if np.max(label) > 255 or np.min(label) < 0:
                bad_files.append((filename, np.unique(label).tolist()))
                
        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {str(e)}")
            bad_files.append((filename, "读取失败"))
            continue
            
    return label_counter, bad_files

def print_report(counter, bad_files):
    """打印统计报告"""
    print("\n" + "="*40)
    print("标签分布报告")
    print("="*40)
    
    # 打印所有存在的标签值
    print("\n[标签值统计]")
    sorted_labels = sorted(counter.items(), key=lambda x: x[0])
    for val, cnt in sorted_labels:
        print(f"  Label {val}: 出现 {cnt} 次 ({cnt/1e6:.2f}M 像素)")

    # 打印异常文件
    if bad_files:
        print("\n[异常文件清单]")
        for filename, bad_vals in bad_files:
            print(f"  {filename}: 异常值 → {bad_vals}")
    else:
        print("\n未发现异常标签值")

    # 统计摘要
    print("\n[关键指标]")
    max_val = max(counter.keys()) if counter else 0
    min_val = min(counter.keys()) if counter else 0
    print(f"最大标签值: {max_val}")
    print(f"最小标签值: {min_val}")
    print(f"唯一标签数: {len(counter)}")

if __name__ == "__main__":
    # 配置路径
    dataset_root = "E:/code/TinySeg/TinySeg_data"  # 修改为你的数据集路径
    anno_dir = os.path.join(dataset_root, "Annotations")
    
    # 运行分析
    counter, bad_files = analyze_labels(anno_dir)
    
    # 生成报告
    print_report(counter, bad_files)
    
    # 保存结果到文件
    with open("label_report.txt", "w") as f:
        f.write("Label Value\tPixel Count\n")
        for val in sorted(counter.keys()):
            f.write(f"{val}\t{counter[val]}\n")
    print("\n报告已保存到 label_report.txt")