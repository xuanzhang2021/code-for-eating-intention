import numpy as np


# 将数据处理成：landmark(N, C, T, V, M) pose(N, T, 3)
def landmark_pose_initial(landmark, pose, label):

    # 参数设置
    sequence_length = 10
    num_sequences = 29948 // sequence_length  # 59

    # 截取可整除的部分
    landmark = landmark[:num_sequences * sequence_length]  # (1947, 68, 2)
    print('landmark_shape_T', landmark.shape)
    pose = pose[:num_sequences * sequence_length]  # (1947, 3)
    print('pose_shape_T', pose.shape)
    label = label[:num_sequences * sequence_length]  # (1947, 3)
    print('label_shape_T', label.shape)

    # 重塑数据为序列形式
    if landmark.shape[-1] == 2:
        # 如果 shape 为 (N, M, 2)
        landmark = landmark.reshape(num_sequences, sequence_length, 68, 2)
    elif landmark.shape[-1] == 3:
        # 如果 shape 为 (N, M, 3)
        landmark = landmark[..., :2]  # 只取最后一维的前两个，从 (N, M, 3) -> (N, M, 2)
        landmark = landmark.reshape(num_sequences, sequence_length, 68, 2)
    else:
        # 如果 shape 不符合预期，抛出错误
        raise ValueError(f"landmarks 的 shape 不正确，期望最后一维为 2 或 3，但实际为 {landmarks.shape[-1]}")

    # pose = pose.reshape(num_sequences, sequence_length, 68, 2)
    label = label.reshape(num_sequences, sequence_length)
    label_last1 = label[:, -1].reshape(-1, 1)  # 从每一行中取出 sequence_length 的最后一个数据

    # 选择19个关键点
    selected_indices = [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54]
    point19_landmarks = landmark[:, :, selected_indices, :]  # (59, 30, 19, 2)

    # 调整数据形状
    point19_landmarks = point19_landmarks.transpose(0, 3, 1, 2)  # (59, 2, 30, 19)
    point19_landmarks = np.expand_dims(point19_landmarks, axis=4)  # (59, 2, 30, 19, 1)

    # 验证数据形状
    print('point19_landmarks reshape:', point19_landmarks.shape)  # (59, 2, 30, 19, 1)
    print('pose reshape:', pose.shape)  # (59, 3)
    print('label reshape:', label.shape)
    print('label_last reshape:', label_last1.shape)

    return point19_landmarks, pose, label, label_last1


def landmark_pose(landmark, pose, label):
    # 参数设置
    sequence_length = 30
    num_samples = landmark.shape[0]
    num_sequences = num_samples - sequence_length + 1  # 滑动窗口生成的序列数量

    # 验证输入数据形状
    print('original landmark shape:', landmark.shape)
    print('original pose shape:', pose.shape)
    print('original label shape:', label.shape)

    # 滑动窗口生成 landmark 序列  shape: (num_sequences, sequence_length, 68, 2)
    landmark_sequences = np.array([landmark[i:i + sequence_length] for i in range(num_sequences)])  #

    # 滑动窗口生成 pose 序列  # shape: (num_sequences, sequence_length, 3)
    pose_sequences = np.array([pose[i:i + sequence_length] for i in range(num_sequences)])

    # 滑动窗口生成 label 序列  # shape: (num_sequences, sequence_length)
    label_sequences = np.array([label[i:i + sequence_length] for i in range(num_sequences)])

    # 提取 label 的最后一个值
    label_last1 = label_sequences[:, -1].reshape(-1, 1)  # shape: (num_sequences, 1)

    # 选择 19 个关键点
    selected_indices = [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54]
    point19_landmarks = landmark_sequences[:, :, selected_indices, :]  # shape: (num_sequences, sequence_length, 19, 2)

    # 调整数据形状
    point19_landmarks = point19_landmarks.transpose(0, 3, 1, 2)  # shape: (num_sequences, 2, sequence_length, 19)
    point19_landmarks = np.expand_dims(point19_landmarks, axis=4)  # shape: (num_sequences, 2, sequence_length, 19, 1)

    # 验证数据形状
    print('point19_landmarks reshape:', point19_landmarks.shape)  # (num_sequences, 2, sequence_length, 19, 1)
    print('pose reshape:', pose_sequences.shape)  # (num_sequences, sequence_length, 3)
    print('label reshape:', label_sequences.shape)  # (num_sequences, sequence_length)
    print('label_last reshape:', label_last1.shape)  # (num_sequences, 1)

    return point19_landmarks, pose_sequences, label_sequences, label_last1


def split_training_test(dataset_input, dataset_label, test_ratio=0.3):
    # 确保输入和标签的样本数量一致
    assert dataset_input.shape[0] == dataset_label.shape[0], "输入数据和标签样本数量不一致！"

    # 获取样本数量
    num_samples = dataset_input.shape[0]

    # 打乱数据索引
    indices = np.arange(num_samples)  # 生成样本索引
    np.random.shuffle(indices)       # 随机打乱索引

    # 根据打乱后的索引重新排列输入数据和标签
    dataset_input = dataset_input[indices]
    dataset_label = dataset_label[indices]

    # 计算分割点
    split_index = int(num_samples * (1 - test_ratio))  # 训练集的分割点

    # 按照分割点划分输入数据和标签
    training_input = dataset_input[:split_index]  # 前 70% 的输入数据
    testing_input = dataset_input[split_index:]   # 后 30% 的输入数据
    training_label = dataset_label[:split_index]  # 前 70% 的标签数据
    testing_label = dataset_label[split_index:]   # 后 30% 的标签数据

    return training_input, testing_input, training_label, testing_label


# 将landmark(N, C, T, V, M) 第二维度拓展一个维度
def channel_expand(initial_data):
    # 创建一个全为 1，形状与 initial_train 在其他维度匹配的新通道
    one_channel = np.ones_like(initial_data[:, :1, :, :, :])  # 形状：(N, 1, 30, 19, 1)

    # 将 ones_channel 与 initial_train 在第二个维度（通道维度）上进行拼接
    expanded_data = np.concatenate((initial_data, one_channel), axis=1)  # 形状：(N, 3, 30, 19, 1)

    # 验证新数据的形状
    print('expanded_train shape:', expanded_data.shape)  # 输出应为 (N, 3, 30, 19, 1)
    return expanded_data


# 加载数据
landmarks = np.load('gw202406301/landmarks.npy')  # 形状：(N, 68, 2)
poses = np.load('gw202406301/pose.npy')            # 形状：(N, 3)
labels = np.load('gw202406301/labels.npy')

print('Landmarks_shape_original:', landmarks.shape)
print('Landmarks_dtype_original:', landmarks.dtype)

landmarks_19, poses_19, label_19, label_19_last = landmark_pose(landmarks, poses, labels)
# landmarks_19 = channel_expand(landmarks_19)

# 保存数据
np.save('gw202406301/landmarks_19_T30.npy', landmarks_19)
np.save('gw202406301/pose_T30.npy', poses_19)
np.save('gw202406301/label_T30.npy', label_19)
np.save('gw202406301/label_T30_last.npy', label_19_last)

# 分割数据集
landmarks_19_T30_train, landmarks_19_T30_test, label_T30_last_train, label_T30_last_test = split_training_test(landmarks_19, label_19_last)

print('landmarks_19_T30_train:', landmarks_19_T30_train.shape)
print('landmarks_19_T30_test:', landmarks_19_T30_test.shape)
print('label_T30_last_train:', label_T30_last_train.shape)
print('label_T30_last_test:', label_T30_last_test.shape)

np.save('gw202406301/landmarks_19_T30_train.npy', landmarks_19_T30_train)
np.save('gw202406301/landmarks_19_T30_test.npy', landmarks_19_T30_test)
np.save('gw202406301/label_T30_last_train.npy', label_T30_last_train)
np.save('gw202406301/label_T30_last_test.npy', label_T30_last_test)




'''
# 处理序列标签（如果需要每个序列一个标签）
pose_sequence_labels = poses_19[:, 0, :]  # (59, 3)
np.save('point19_train_label_sequence.npy', pose_sequence_labels)
print(pose_sequence_labels.shape)
'''

