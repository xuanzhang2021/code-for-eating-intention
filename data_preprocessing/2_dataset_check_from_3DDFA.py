import os
import numpy as np
import pandas as pd

# 文件路径
csv_file_path = "/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/data/add0/zx202406303.csv"
npy_file_path = "./zx202406301/labels.npy"

# 检查文件是否存在
if not os.path.exists(csv_file_path):
    print(f"CSV文件不存在，请检查路径：{csv_file_path}")
    exit()

if not os.path.exists(npy_file_path):
    print(f"labels.npy 文件不存在，请检查路径：{npy_file_path}")
    exit()

# 加载 CSV 文件中的 label 列
try:
    csv_data = pd.read_csv(csv_file_path)
    if 'eating_timing' not in csv_data.columns:
        raise ValueError(f"CSV文件中未找到'eating_timing'列，请检查文件内容！")

    csv_labels = csv_data['eating_timing'].values  # 提取 label 列为 NumPy 数组
    csv_labels = csv_labels.reshape(-1, 1)  # 调整形状为 (N, 1)
except Exception as e:
    print(f"读取 CSV 文件时发生错误：{e}")
    exit()

# 加载 labels.npy 文件
try:
    npy_labels = np.load(npy_file_path)
except Exception as e:
    print(f"加载 labels.npy 文件时发生错误：{e}")
    exit()

# 对比两个数组是否一致
if np.array_equal(csv_labels, npy_labels):
    print("labels.npy 和 CSV 文件中的 eating_timing 列完全一致！")
else:
    print("labels.npy 和 CSV 文件中的 eating_timing 列不一致！")
    # 打印不一致的具体信息
    print(f"CSV 文件中的 labels（前10行）：\n{csv_labels[:10]}")  # 打印前10行
    print(f"npy 文件中的 labels（前10行）：\n{npy_labels[:10]}")  # 打印前10行
    # 检查形状是否一致
    if csv_labels.shape != npy_labels.shape:
        print(f"形状不一致：CSV eating_timing 形状为 {csv_labels.shape}，npy labels 形状为 {npy_labels.shape}")
    else:
        # 如果形状一致，但内容不一致，检查具体不同的位置
        differences = np.where(csv_labels != npy_labels)
        print(f"不一致的位置索引：{differences}")
        print(f"CSV 文件中对应的不一致值：{csv_labels[differences]}")
        print(f"npy 文件中对应的不一致值：{npy_labels[differences]}")


# 文件路径
folder_path = "/home/zhangxuan/sourcecode/video_to_images/pngs/zx202406301/"
landmarks_file = "./zx202406301/landmarks.npy"
pose_file = "./zx202406301/pose.npy"

# 检查 landmarks.npy 和 pose.npy 是否存在
if not os.path.exists(landmarks_file):
    print(f"landmarks.npy 文件不存在，请检查路径：{landmarks_file}")
    exit()

if not os.path.exists(pose_file):
    print(f"pose.npy 文件不存在，请检查路径：{pose_file}")
    exit()

# 加载 landmarks.npy 和 pose.npy
try:
    landmarks = np.load(landmarks_file)  # Shape: (N, 68, 3)
    poses = np.load(pose_file)  # Shape: (N, P)
except Exception as e:
    print(f"加载 landmarks.npy 或 pose.npy 时发生错误：{e}")
    exit()

# 获取所有 txt 文件，并按时间戳排序
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
txt_files.sort(key=lambda x: float(x.split('_')[0]))  # 按时间戳排序

# 检查 landmarks.npy 与 txt 文件第 1、2、3 行数据是否一致
all_landmarks_match = True
all_poses_match = True

for idx, txt_file in enumerate(txt_files):
    txt_path = os.path.join(folder_path, txt_file)

    # 打开并读取 txt 文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

        # 确保文件至少有 4 行
        if len(lines) < 4:
            print(f"文件 {txt_file} 行数不足，跳过...")
            continue

        # 从 txt 文件中提取 1、2、3 行（landmarks）和第 4 行（pose）
        txt_landmarks = np.array([list(map(float, line.strip().split())) for line in lines[:3]])  # Shape: (3, 68)
        txt_pose = np.array(list(map(float, lines[3].strip().split())))  # Shape: (P,)

        # 对比 landmarks
        npy_landmarks = landmarks[idx]  # Shape: (68, 3)
        txt_landmarks_transposed = np.transpose(txt_landmarks, (1, 0))  # Shape: (68, 3)

        if not np.allclose(npy_landmarks, txt_landmarks_transposed, atol=1e-6):
            print(f"landmarks 不一致！文件: {txt_file}")
            print(f"npy_landmarks (前5个点):\n{npy_landmarks[:5]}")
            print(f"txt_landmarks_transposed (前5个点):\n{txt_landmarks_transposed[:5]}")
            all_landmarks_match = False

        # 对比 pose
        npy_pose = poses[idx]  # Shape: (P,)
        if not np.allclose(npy_pose, txt_pose, atol=1e-6):
            print(f"pose 不一致！文件: {txt_file}")
            print(f"npy_pose:\n{npy_pose}")
            print(f"txt_pose:\n{txt_pose}")
            all_poses_match = False

# 总结是否一致
if all_landmarks_match:
    print("landmarks.npy 和所有 txt 文件的第 1、2、3 行数据完全一致！")
else:
    print("landmarks.npy 和部分 txt 文件的第 1、2、3 行数据不一致！")

if all_poses_match:
    print("pose.npy 和所有 txt 文件的第 4 行数据完全一致！")
else:
    print("pose.npy 和部分 txt 文件的第 4 行数据不一致！")


data_label = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/zx202406301/labels.npy')
data_landmark = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/zx202406301/landmarks.npy')
data_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/zx202406301/pose.npy')

print('num_data_label', data_label.shape)
print('num_data_landmarks', data_landmark.shape)
print('num_data_pose', data_pose.shape)
