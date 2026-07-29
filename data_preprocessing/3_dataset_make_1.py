import numpy as np


# 制作channel 为 3的数据集
def landmark_pose(landmark, pose, label):
    # 参数设置
    sequence_length = 30
    num_samples = landmark.shape[0]
    num_sequences = num_samples - sequence_length + 1  # 滑动窗口生成的序列数量

    poses_expanded = np.expand_dims(poses, axis=1)  # 或 poses[:, np.newaxis, :]

    # 拼接 landmarks 和 poses
    landmark_pose = np.concatenate((landmarks, poses_expanded), axis=1)  # (15049, 69, 3)

    # 滑动窗口生成 landmark 序列  shape: (num_sequences, sequence_length, 68, 3)
    landmark_pose_sequences = np.array([landmark_pose[i:i + sequence_length] for i in range(num_sequences)])  #

    # 滑动窗口生成 label 序列  # shape: (num_sequences, sequence_length)
    label_sequences = np.array([label[i:i + sequence_length] for i in range(num_sequences)])

    # 提取 label 的最后一个值
    label_last1 = label_sequences[:, -1].reshape(-1, 1)  # shape: (num_sequences, 1)

    # 选择 19 个关键点
    selected_indices = [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54, 68]
    point20_landmarks_pose = landmark_pose_sequences[:, :, selected_indices, :]  # shape: (num_sequences, sequence_length, 19, 2)

    # 调整数据形状
    point20_landmarks_pose = point20_landmarks_pose.transpose(0, 3, 1, 2)  # shape: (num_sequences, 2, sequence_length, 19)
    point20_landmarks_pose = np.expand_dims(point20_landmarks_pose, axis=4)  # shape: (num_sequences, 2, sequence_length, 19, 1)


    return point20_landmarks_pose, label_sequences, label_last1


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


def split_training_valid_test(dataset_input, dataset_label, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    # 确保输入和标签的样本数量一致
    assert dataset_input.shape[0] == dataset_label.shape[0], "输入数据和标签样本数量不一致！"
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), "比例之和必须为 1！"

    # 获取样本数量
    num_samples = dataset_input.shape[0]

    # 打乱数据索引
    indices = np.arange(num_samples)  # 生成样本索引
    np.random.shuffle(indices)       # 随机打乱索引

    # 根据打乱后的索引重新排列输入数据和标签
    dataset_input = dataset_input[indices]
    dataset_label = dataset_label[indices]

    # 计算分割点
    train_split_index = int(num_samples * train_ratio)  # 训练集的分割点
    valid_split_index = int(num_samples * (train_ratio + valid_ratio))  # 验证集的分割点

    # 按照分割点划分输入数据和标签
    training_input = dataset_input[:train_split_index]  # 前 train_ratio 的输入数据
    validation_input = dataset_input[train_split_index:valid_split_index]  # 中间 valid_ratio 的输入数据
    testing_input = dataset_input[valid_split_index:]   # 剩余 test_ratio 的输入数据

    training_label = dataset_label[:train_split_index]  # 前 train_ratio 的标签数据
    validation_label = dataset_label[train_split_index:valid_split_index]  # 中间 valid_ratio 的标签数据
    testing_label = dataset_label[valid_split_index:]   # 剩余 test_ratio 的标签数据

    return training_input, validation_input, testing_input, training_label, validation_label, testing_label


# 加载数据
landmarks = np.load('gw202406301/landmarks.npy')  # 形状：(N, 68, 3)
poses = np.load('gw202406301/pose.npy')            # 形状：(N, 3)
labels = np.load('gw202406301/labels.npy')

# 验证输入数据形状
print('original landmark shape:', landmarks.shape)
print('original pose shape:', poses.shape)
print('original label shape:', labels.shape)


landmarks_pose_20, label_20, label_20_last = landmark_pose(landmarks, poses, labels)
# landmarks_19 = channel_expand(landmarks_19)

# 验证数据形状
print('point20_landmarks reshape:', landmarks_pose_20.shape)  # (num_sequences, 2, sequence_length, 19, 1)
print('label reshape:', label_20.shape)  # (num_sequences, sequence_length)
print('label_last reshape:', label_20_last.shape)  # (num_sequences, 1)

# 分割数据集
landmarks_20_T30_train, landmarks_20_T30_valid, landmarks_20_T30_test, label_T30_last_train, label_T30_last_valid, label_T30_last_test = split_training_valid_test(landmarks_pose_20, label_20_last)

print('landmarks_20_T30_train:', landmarks_20_T30_train.shape)
print('landmarks_20_T30_test:', landmarks_20_T30_test.shape)
print('label_T30_last_train_20:', label_T30_last_train.shape)
print('label_T30_last_test_20:', label_T30_last_test.shape)


# 保存数据
# np.save('gw202406301/landmarks_20_T30.npy', landmarks_pose_20)
# np.save('gw202406301/label_T30_20.npy', label_20)
# np.save('gw202406301/label_T30_last_20.npy', label_20_last)

np.save('gw202406301/landmarks_20_T30_train.npy', landmarks_20_T30_train)
np.save('gw202406301/landmarks_20_T30_valid.npy', landmarks_20_T30_valid)
np.save('gw202406301/landmarks_20_T30_test.npy', landmarks_20_T30_test)
np.save('gw202406301/label_20_T30_last_train.npy', label_T30_last_train)
np.save('gw202406301/label_20_T30_last_valid.npy', label_T30_last_valid)
np.save('gw202406301/label_20_T30_last_test.npy', label_T30_last_test)






