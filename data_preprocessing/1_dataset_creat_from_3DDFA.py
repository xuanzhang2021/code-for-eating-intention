import os
import numpy as np
import pandas as pd

# 设置文件夹路径和保存路径,读取TXT文件
folder_path = "/home/zhangxuan/sourcecode/video_to_images/pngs/zx202406301/"
# 获取label数据所在文件的路径
csv_file_path = "/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/data/add0/zx202406303.csv"
save_path = "./zx202406301/"

# 创建保存路径（如果不存在）
os.makedirs(save_path, exist_ok=True)


# 检查文件数量是否相等
def check_file_counts(folder_path):
    # 获取所有文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # 检查数量是否相等
    if len(png_files) != len(txt_files) or len(txt_files) != len(jpg_files):
        raise ValueError(
            f"文件数量不匹配：PNG文件数量={len(png_files)}, TXT文件数量={len(txt_files)}, JPG文件数量={len(jpg_files)}。请确保文件数量一致后再运行。"
        )


# 调用检查文件数量函数
try:
    check_file_counts(folder_path)
except Exception as e:
    print(f"文件数量检查失败：{e}")
    exit()

# 获取所有txt文件，并按照时间戳排序
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
txt_files.sort(key=lambda x: float(x.split('_')[0]))

# 初始化存储列表
landmarks = []
poses = []

# 逐个读取txt文件
for txt_file in txt_files:
    txt_path = os.path.join(folder_path, txt_file)

    # 打开并读取txt文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

        # 确保文件至少有4行
        if len(lines) < 4:
            print(f"文件 {txt_file} 行数不足，跳过...")
            continue

        # 提取前三行（landmarks）并转换为float
        landmark = [list(map(float, line.strip().split())) for line in lines[:3]]
        landmarks.append(landmark)

        # 提取第4行（pose）并转换为float
        pose = list(map(float, lines[3].strip().split()))
        poses.append(pose)

# 转换为numpy数组
landmarks = np.array(landmarks)  # 原始形状为 (N, 3, 68)
poses = np.array(poses)

# 调整 landmarks 的形状为 (N, 68, 3)
landmarks = np.transpose(landmarks, (0, 2, 1))  # 交换轴 1 和轴 2

# 保存 landmarks 和 poses 到 .npy 文件
try:
    np.save(os.path.join(save_path, "landmarks.npy"), landmarks)
    np.save(os.path.join(save_path, "pose.npy"), poses)
    print("landmarks.npy 和 pose.npy 文件已成功保存！")
except Exception as e:
    print(f"保存 landmarks.npy 和 pose.npy 时发生错误：{e}")
    exit()

# 读取CSV文件并提取 "label" 列
try:
    # 使用 pandas 读取 CSV 文件
    data = pd.read_csv(csv_file_path)

    # 检查是否存在 "label" 列
    if 'eating_timing' not in data.columns:
        raise ValueError(f"CSV文件中未找到'label'列，请检查文件内容！")

    # 提取 "label" 列数据
    labels = data['eating_timing'].values  # 提取为 NumPy 数组
    labels = labels.reshape(-1, 1)  # 将数据形状调整为 (N, 1)

    # 保存为 .npy 文件
    np.save(os.path.join(save_path, "labels.npy"), labels)
    print(f"labels.npy 文件已成功保存到路径: {save_path}")

except FileNotFoundError:
    print(f"CSV文件未找到，请检查路径是否正确：{csv_file_path}")
    exit()
except Exception as e:
    print(f"处理 CSV 文件时发生错误：{e}")
    exit()
