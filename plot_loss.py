import face_alignment
import numpy as np
from math import atan2, sqrt, pi
from skimage import io
import os

data_image1 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/landmarks_19_T30_gwzx_train.npy')
data_image_size1 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/landmarks_input.npy')

print(data_image1.shape)
print(data_image_size1.shape)

import matplotlib.pyplot as plt

# 文件路径
file_paths = [
    'optimization_result/validation_loss_log1.txt',
    'optimization_result/validation_loss_log2.txt',
    'optimization_result/validation_loss_log3_2.txt',
    'optimization_result/validation_loss_log4.txt',
    'optimization_result/training_loss_log1.txt',
    'optimization_result/training_loss_log2.txt',
    'optimization_result/training_loss_log3_2.txt',
    'optimization_result/training_loss_log4.txt',
]

# 文件对应的标签
labels = [
    'Validation Loss of NAG',
    'Validation Loss of SGD',
    'Validation Loss of GDM',
    'Validation Loss of Adam',
    'Training Loss of NAG',
    'Training Loss of SGD',
    'Training Loss of GDM',
    'Training Loss of Adam',
]

# 颜色列表，为每条曲线指定不同的颜色
colors = [
    'orange', 'red', 'blue', 'green',
    'purple', 'brown', 'pink', 'cyan'
]

# 读取所有文件内容
losses = []
max_epochs = 0  # 用于记录最长的 epochs 数

for file_path in file_paths:
    with open(file_path, 'r') as f:
        loss = [float(line.strip()) for line in f]
        losses.append(loss)
        max_epochs = max(max_epochs, len(loss))  # 更新最长的 epochs 数

# 绘制所有曲线
plt.figure(figsize=(12, 8))
for i, loss in enumerate(losses):
    epochs = range(1, len(loss) + 1)  # 根据每个文件的行数生成 epochs
    plt.plot(epochs, loss, label=labels[i], color=colors[i], marker='o')

# 添加标题和标签
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 显示图像
plt.show()



'''

def modify_npy_shape(input_path, output_path, target_channels):
    # 1. 加载原始数据
    data = np.load(input_path)
    print(f"Original shape: {data.shape}")

    # 2. 检查数据形状
    if data.shape[1] < max(target_channels) + 1:
        raise ValueError(f"Invalid target_channels {target_channels}. "
                         f"Input data has only {data.shape[1]} channels.")

    # 3. 裁剪第 2 维度（通道）
    data_modified = data[:, target_channels, :, :, :]
    print(f"Modified shape: {data_modified.shape}")

    # 4. 保存修改后的数据
    np.save(output_path, data_modified)
    print(f"Modified data saved to: {output_path}")


# 定义输入文件路径和输出文件路径
input_path = 'data_preprocessing/gw202406301/landmarks_19_T30_gwzx_train.npy'
output_path = 'data_preprocessing/gw202406301/landmarks_19_T30_gwzx_train_2D.npy'

# 调用函数，保留第 0 和第 1 通道
modify_npy_shape(input_path, output_path, target_channels=[0, 1])

input_path = 'data_preprocessing/gw202406301/landmarks_19_T30_gwzx_test.npy'
output_path = 'data_preprocessing/gw202406301/landmarks_19_T30_gwzx_test_2D.npy'

# 调用函数，保留第 0 和第 1 通道
modify_npy_shape(input_path, output_path, target_channels=[0, 1])




def merge_npy_files(file1_path, file2_path, output_path):
    # 加载第一个 .npy 文件
    if os.path.exists(file1_path):
        data1 = np.load(file1_path)
        print(f"加载文件: {file1_path}, 数据形状: {data1.shape}")
    else:
        raise FileNotFoundError(f"文件 {file1_path} 不存在！")

    # 加载第二个 .npy 文件
    if os.path.exists(file2_path):
        data2 = np.load(file2_path)
        print(f"加载文件: {file2_path}, 数据形状: {data2.shape}")
    else:
        raise FileNotFoundError(f"文件 {file2_path} 不存在！")

    # 拼接两个数组
    merged_data = np.concatenate((data1, data2), axis=0)  # 按第一维度拼接
    print(f"合并后的数据形状: {merged_data.shape}")

    # 保存为新的 .npy 文件
    np.save(output_path, merged_data)
    print(f"合并后的数据已保存到: {output_path}")


# 文件路径
file1_path = "data_preprocessing/gw202406301/landmarks_19_T30_train.npy"  # 文件夹1中的 landmarks.npy 文件路径
file2_path = "data_preprocessing/zx202406301/landmarks_19_T30_train.npy"  # 文件夹1中的 landmarks.npy 文件路径
output_path = "data_preprocessing/gw202406301/landmarks_19_T30_gwzx_train.npy"  # 合并后的 .npy 文件保存路径

# 调用函数进行合并
merge_npy_files(file1_path, file2_path, output_path)

# 文件路径
file1_path = "data_preprocessing/gw202406301/landmarks_19_T30_test.npy"  # 文件夹1中的 landmarks.npy 文件路径
file2_path = "data_preprocessing/zx202406301/landmarks_19_T30_test.npy"  # 文件夹1中的 landmarks.npy 文件路径
output_path = "data_preprocessing/gw202406301/landmarks_19_T30_gwzx_test.npy"  # 合并后的 .npy 文件保存路径

# 调用函数进行合并
merge_npy_files(file1_path, file2_path, output_path)

file1_path = "data_preprocessing/gw202406301/label_T30_last_train.npy"  # 文件夹2中的 landmarks.npy 文件路径
file2_path = "data_preprocessing/zx202406301/label_T30_last_train.npy"  # 文件夹2中的 landmarks.npy 文件路径
output_path = "data_preprocessing/gw202406301/label_T30_last_gwzx_train.npy"  # 合并后的 .npy 文件保存路径

merge_npy_files(file1_path, file2_path, output_path)

file1_path = "data_preprocessing/gw202406301/label_T30_last_test.npy"  # 文件夹2中的 landmarks.npy 文件路径
file2_path = "data_preprocessing/zx202406301/label_T30_last_test.npy"  # 文件夹2中的 landmarks.npy 文件路径
output_path = "data_preprocessing/gw202406301/label_T30_last_gwzx_test.npy"  # 合并后的 .npy 文件保存路径

merge_npy_files(file1_path, file2_path, output_path)


data_image = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/BIWI_gw20240630_3D/image.npy')
data_image_size = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/BIWI_gw20240630_3D/img_size.npy')
data_landmark = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/BIWI_gw20240630_3D/landmark.npy')
data_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/BIWI_gw20240630_3D/pose.npy')

print(data_image.shape)
print(data_image_size.shape)
print(data_landmark.shape)
# print(data_landmark)
print(data_pose.shape)

data_image4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/image.npy')
data_image_size4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/img_size.npy')
data_landmark4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/landmark.npy')
data_pose4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/pose.npy')
print(data_image4.shape)
print(data_image_size4.shape)
print(data_landmark4.shape)
# print(data_landmark)
print(data_pose4.shape)

import os
import shutil

def merge_folders(folder1, folder2, output_folder):
    # 检查源文件夹是否存在
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print("源文件夹不存在，请检查路径！")
        return

    # 创建目标文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 合并第一个文件夹
    merge_images_from_folder(folder1, output_folder)

    # 合并第二个文件夹
    merge_images_from_folder(folder2, output_folder)

    print(f"已成功将 {folder1} 和 {folder2} 合并到 {output_folder} 中！")


def merge_images_from_folder(source_folder, target_folder):
    """
    将 source_folder 中的图片移动到 target_folder 中。
    如果存在重复文件名，则自动重命名文件。
    """
    for image in os.listdir(source_folder):
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(target_folder, image)

        # 如果是文件而不是文件夹
        if os.path.isfile(src_path):
            if os.path.exists(dst_path):
                # 如果目标文件夹中已存在同名文件，重命名文件
                dst_path = resolve_duplicate_name(target_folder, image)

            shutil.move(src_path, dst_path)  # 移动文件
            print(f"已移动文件：{src_path} -> {dst_path}")


def resolve_duplicate_name(folder, filename):
    """
    如果文件名重复，则生成一个新的文件名。
    假设文件名格式为 'name.ext'，例如 'image.jpg'。
    """
    name, ext = os.path.splitext(filename)
    counter = 1

    # 生成新文件名，直到不存在重复
    new_filename = f"{name}_{counter}{ext}"
    while os.path.exists(os.path.join(folder, new_filename)):
        counter += 1
        new_filename = f"{name}_{counter}{ext}"

    return os.path.join(folder, new_filename)


def count_files_in_folder(folder):
    """
    统计文件夹中的文件数量（不包括子文件夹）。
    """
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


# 示例：运行代码
folder_1 = "/home/zhangxuan/sourcecode/video_to_images/pngs/gw20240630/1"  # 第一个文件夹路径
folder_2 = "/home/zhangxuan/sourcecode/video_to_images/pngs/gw20240630/2"  # 第二个文件夹路径
merged_output_folder = "/home/zhangxuan/sourcecode/video_to_images/pngs/gw20240630"  # 合并后的目标文件夹路径

# merge_folders(folder_1, folder_2, merged_output_folder)
print(count_files_in_folder(merged_output_folder))




data_image1 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/landmarks_input.npy')
data_image_size1 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/pose_label.npy')
data_label = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/labels.npy')


print(data_image1.shape)
print(data_image_size1.shape)
print(data_label.shape)


def estimate_pose_from_landmarks(landmarks_3d):
    """
    根据 3D 人脸关键点计算头部姿态（roll, pitch, yaw）。
    """
    # 选择一些关键点的索引（以 68 个关键点为基础）
    # 鼻尖: 30, 左眼左角: 36, 右眼右角: 45, 左嘴角: 48, 右嘴角: 54, 下巴: 8
    nose_tip = landmarks_3d[30]
    chin = landmarks_3d[8]
    left_eye_corner = landmarks_3d[36]
    right_eye_corner = landmarks_3d[45]

    # 定义头部方向向量
    forward_vector = nose_tip - chin  # 鼻尖到下巴的向量
    left_vector = left_eye_corner - right_eye_corner  # 左眼到右眼的向量

    # 计算 roll（绕 Z 轴的旋转）
    roll = -atan2(forward_vector[1], forward_vector[2]) * 180 / pi

    # 计算 pitch（绕 X 轴的旋转）
    pitch = atan2(forward_vector[0], sqrt(forward_vector[1]**2 + forward_vector[2]**2)) * 180 / pi

    # 计算 yaw（绕 Y 轴的旋转）
    yaw = atan2(left_vector[1], left_vector[0]) * 180 / pi

    return roll, pitch, yaw


# 初始化 face_alignment 对象，选择 3D 模式
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

# 加载输入图像
image_path = '/home/zhangxuan/sourcecode/EVA-GCN-main/data/BIWI/1/950.png'  # 替换为你的图片路径
input_image = io.imread(image_path)

# 检测 3D 人脸关键点
preds = fa.get_landmarks(input_image)

if preds:
    # 如果检测到人脸，处理第一张人脸
    landmarks_3d = preds[0]  # 3D 关键点，(68, 3)，每个关键点包含 (x, y, z)

    # 输出关键点
    print("3D Landmarks:")
    print(landmarks_3d)

    # 估计姿态
    roll, pitch, yaw = estimate_pose_from_landmarks(landmarks_3d)

    # 输出姿态信息
    print(f"Head Pose:")
    print(f"  Roll: {roll:.2f}°")
    print(f"  Pitch: {pitch:.2f}°")
    print(f"  Yaw: {yaw:.2f}°")
else:
    print("No face detected!")


data_image = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/AFW/image.npy')
data_image_size = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/AFW/img_size.npy')
data_landmark = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/AFW/landmark.npy')
data_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/AFW/pose.npy')

print(data_image.shape)
print(data_image_size.shape)
print(data_landmark.shape)
print(data_pose.shape)


data_image1 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW/image.npy')
data_image_size1 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW/img_size.npy')
data_landmark1 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW/landmark.npy')
data_pose1 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW/pose.npy')

print(data_image1.shape)
print(data_image_size1.shape)
print(data_landmark1.shape)
print(data_pose1.shape)


data_image2 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW_Flip/image.npy')
data_image_size2 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW_Flip/img_size.npy')
data_landmark2 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW_Flip/landmark.npy')
data_pose2 = np.load('/home/zhangxuan/Downloads/data.zip/data/AFW_Flip/pose.npy')

data_image3 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_test/image.npy')
data_image_size3 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_test/img_size.npy')
data_landmark3 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_test/landmark.npy')
data_pose3 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_test/pose.npy')

data_image4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/image.npy')
data_image_size4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/img_size.npy')
data_landmark4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/landmark.npy')
data_pose4 = np.load('/home/zhangxuan/Downloads/data.zip/data/BIWI_train/pose.npy')

# data1 = np.load('/media/zhangxuan/KINGSTON/NTU-RGB-D/train_label.pkl')

# data2 = np.load('/media/zhangxuan/KINGSTON/NTU-RGB-D/val_data.npy')
# data3 = np.load('/media/zhangxuan/KINGSTON/NTU-RGB-D/val_label.pkl')


print(data_image1.shape)
print(data_image_size1.shape)
print(data_landmark1.shape)
print(data_pose1.shape)

print(data_image2.shape)
print(data_image_size2.shape)
print(data_landmark2.shape)
print(data_pose2.shape)

print(data_image3.shape)
print(data_image_size3.shape)
print(data_landmark3.shape)
print(data_pose3.shape)

print(data_image4.shape)
print(data_image_size4.shape)
print(data_landmark4.shape)
print(data_pose4.shape)
# print(data1)
# print(data2.shape)
'''