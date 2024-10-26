import math
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
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

# 检查是否有可用的 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
model = Tudui().to(device)
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
        X_data, Y_data = X_data.to(device), Y_data.to(device)  # 将数据移动到 GPU
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
                X_test_data, Y_test_data = X_test_data.to(device), Y_test_data.to(device)  # 将数据移动到 GPU
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