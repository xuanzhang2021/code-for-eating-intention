import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from net.st_gcn import Model

# 这个代码其实可以不需要，把processor的train改成test就可以实现test数据集上的evaluation
# 自定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)  # 加载 landmarks 数据
        self.labels = np.load(label_path)  # 加载 ground truth 标签
        assert self.data.shape[0] == self.labels.shape[0], "数据和标签数量不一致！"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # landmarks 数据
        y = self.labels[idx]  # 二分类标签（0 或 1）
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# 测试函数
def test_model(model_path, test_data_path, test_label_path, batch_size=32):
    # 加载测试数据
    test_dataset = TestDataset(test_data_path, test_label_path)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 替换为实际的模型定义类名，例如 YourModelClass
    model = Model(
        in_channels=2,  # 输入通道数，例如 3 表示 (x, y, z) 坐标
        num_class=2,  # 二分类问题
        graph_args={'layout': 'openpose_face_19_points', 'strategy': 'spatial'},  # 示例图参数
        edge_importance_weighting=True  # 启用边权重学习
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式

    all_predictions = []
    all_labels = []

    # 遍历测试集
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # 模型预测
            outputs = model(batch_data)  # 假设输出为 [batch_size, 2] 的 logits
            predicted_classes = torch.argmax(outputs, dim=1)  # 获取预测类别
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
        print(all_predictions)
        print(all_labels)

    # 计算二分类评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    print(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


# 测试模型
def main():
    model_path = "/home/zhangxuan/sourcecode/EVA-GCN-main/work_dir_19_points_FAN_6_layers_train_on_300W_LP_test_on_BIWI/GCN_head_pose/epoch107_model_mae_4.076589107513428.pt/epoch299_model.pt"
    test_data_path = "/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/landmarks_19_T30_gwzx_test_2D_1.npy"
    test_label_path = "/home/zhangxuan/sourcecode/EVA-GCN-main/data_preprocessing/gw202406301/label_T30_last_gwzx_test_1.npy"

    print("Calling test_model with:", model_path, test_data_path, test_label_path)
    test_model(model_path, test_data_path, test_label_path, batch_size=32)

if __name__ == "__main__":
    print(111111111)
    main()
