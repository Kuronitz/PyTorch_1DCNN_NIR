import math
import warnings

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 忽略运行时警告
warnings.simplefilter(action='ignore', category=RuntimeWarning)
# 设置随机种子以确保可重复性
torch.manual_seed(2022)

class Tudui(nn.Module):
    """
    继承自 nn.Module 的神经网络模型类。
    """
    def __init__(self):
        """
        使用两个顺序子模型初始化 Tudui 模型。
        """
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(1, 16, 2),  # 1D 卷积层
            nn.ReLU(),            # ReLU 激活函数
            nn.MaxPool1d(2),      # 最大池化层
            nn.Conv1d(16, 32, 2), # 1D 卷积层
            nn.ReLU(),            # ReLU 激活函数
            nn.MaxPool1d(4),      # 最大池化层
            nn.Flatten()          # 展平层
        )
        self.model2 = nn.Sequential(
            nn.Linear(32, 2),     # 全连接层
            nn.Sigmoid()          # Sigmoid 激活函数
        )

    def forward(self, input):
        """
        模型的前向传播。

        参数:
            input (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过模型处理后的输出张量。
        """
        input = input.reshape(-1, 1, 11)  # 重塑输入张量
        x = self.model1(input)            # 通过第一个子模型
        x = self.model2(x)                # 通过第二个子模型
        return x

def load_data(file_path):
    """
    从 CSV 文件加载数据并进行预处理。

    参数:
        file_path (str): CSV 文件路径。

    返回:
        tuple: 包含特征张量和标签张量的元组。
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:12].values.astype(float)  # 提取特征
    Y = LabelEncoder().fit_transform(data.iloc[:, 0].values)  # 编码标签
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 加载训练和测试数据
X_train, Y_train = load_data("./train.csv")
X_test, Y_test = load_data("./test.csv")

# 为训练和测试数据创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=4444, shuffle=True)

# 初始化模型、损失函数和优化器
model = Tudui()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000  # 训练轮数

# 训练循环
for epoch in range(epochs):
    print(f"--------第 {epoch + 1} 轮训练--------")
    total_G_mean = 0
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model.train()  # 设置模型为训练模式

    for X_data, Y_data in train_loader:
        output = model(X_data)  # 前向传播
        loss = loss_function(output, Y_data)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        pred = output.argmax(axis=1)  # 获取预测结果
        matrix = confusion_matrix(Y_data, pred)  # 计算混淆矩阵
        TN, FP, FN, TP = matrix.ravel()  # 提取混淆矩阵元素
        FDR = TP / (TP + FN)  # 计算假发现率
        P = TN / (TN + FP)  # 计算精度
        G_mean = math.sqrt(FDR * P) if not np.isnan(FDR * P) else 0.0  # 计算 G-mean
        total_G_mean += G_mean

    print(f"G-mean: {total_G_mean / len(train_loader):.4f}")

    if (epoch + 1) % 10 == 0:
        total_G_mean_test = 0
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            for X_test_data, Y_test_data in test_loader:
                out = model(X_test_data)  # 前向传播
                pred_test = out.argmax(axis=1)  # 获取预测结果
                matrix = confusion_matrix(Y_test_data, pred_test)  # 计算混淆矩阵
                TN, FP, FN, TP = matrix.ravel()  # 提取混淆矩阵元素
                FDR = TP / (TP + FN)  # 计算假发现率
                P = TN / (TN + FP)  # 计算精度
                G_mean_test = math.sqrt(FDR * P) if not np.isnan(FDR * P) else 0.0  # 计算 G-mean
                total_G_mean_test += G_mean_test

        print("**********************验证***********************")
        print(f"测试集上的 G-mean: {total_G_mean_test / len(test_loader):.4f}")

    import unittest
    import torch
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset


    class TestTuduiModel(unittest.TestCase):

        def setUp(self):
            self.model = Tudui()
            self.loss_function = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.X_train, self.Y_train = load_data("./train.csv")
            self.X_test, self.Y_test = load_data("./test.csv")
            self.train_dataset = TensorDataset(self.X_train, self.Y_train)
            self.test_dataset = TensorDataset(self.X_test, self.Y_test)
            self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=4444, shuffle=True)

        def test_forward_pass(self):
            input_tensor = torch.randn(1, 11)
            output = self.model(input_tensor)
            self.assertEqual(output.shape, (1, 2))

        def test_training_step(self):
            for X_data, Y_data in self.train_loader:
                output = self.model(X_data)
                loss = self.loss_function(output, Y_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.assertGreaterEqual(loss.item(), 0)

        def test_load_data(self):
            X, Y = load_data("./train.csv")
            self.assertEqual(X.shape[1], 11)
            self.assertEqual(len(X), len(Y))

        def test_g_mean_calculation(self):
            for X_data, Y_data in self.train_loader:
                output = self.model(X_data)
                pred = output.argmax(axis=1)
                matrix = confusion_matrix(Y_data, pred)
                TN, FP, FN, TP = matrix.ravel()
                FDR = TP / (TP + FN)
                P = TN / (TN + FP)
                G_mean = math.sqrt(FDR * P) if not np.isnan(FDR * P) else 0.0
                self.assertGreaterEqual(G_mean, 0.0)
                self.assertLessEqual(G_mean, 1.0)

        def test_validation_step(self):
            total_G_mean_test = 0
            self.model.eval()
            with torch.no_grad():
                for X_test_data, Y_test_data in self.test_loader:
                    out = self.model(X_test_data)
                    pred_test = out.argmax(axis=1)
                    matrix = confusion_matrix(Y_test_data, pred_test)
                    TN, FP, FN, TP = matrix.ravel()
                    FDR = TP / (TP + FN)
                    P = TN / (TN + FP)
                    G_mean_test = math.sqrt(FDR * P) if not np.isnan(FDR * P) else 0.0
                    total_G_mean_test += G_mean_test
            self.assertGreaterEqual(total_G_mean_test / len(self.test_loader), 0.0)
            self.assertLessEqual(total_G_mean_test / len(self.test_loader), 1.0)


    if __name__ == '__main__':
        unittest.main()