# 导包
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
import time
from torch.utils.data import TensorDataset
import random
from torch.utils.data import DataLoader, Subset
# 配置GPU，这里有两种方式
## 方案一：使用os.environ
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 64
num_workers = 0   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
lr = 1e-1
epochs = 150



image_size = 28

# class CustomDataset(Dataset):
#     def __init__(self, npz_file):
#         self.data = np.load(npz_file)
#         self.x = self.data["features"]
#         self.y = self.data["labels"]
#         # self.x = torch.from_numpy(self.x).float()
#         # self.y = torch.from_numpy(self.y).long()
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
#
# dataset = CustomDataset('./purchase/purchase20.npz')
# print(len(dataset))
# # 定义训练集和测试集的样本数
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
#
# # 划分训练集和测试集
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# 加载.npz数据集
data = np.load('./purchase/purchase20.npz')
x, y = data["features"], data["labels"]
y=np.argmax(y, axis=1)

# 将数据转换成Tensor格式
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).long()

# 创建TensorDataset和DataLoader
dataset = TensorDataset(x, y)

indices = random.sample(range(len(dataset)), int(len(dataset)*0.8))
# train_data = Subset(dataset,  [i for i in range(0, 38758)])
train_data = Subset(dataset,  [i for i in range(0, 15000)])
print(len(train_data))
# test_data = Subset(dataset, [i for i in range(38759, 48448)])
test_data = Subset(dataset, [i for i in range(15000, 18000)])
print(len(test_data))
# 定义DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class MLPNet(nn.Module):
    def __init__(self, size):
        super(MLPNet, self).__init__()
        self.size = size
        self.layers = nn.ModuleList()
        for i in range(len(size)-1):
            self.layers.append(nn.Linear(size[i], size[i+1]))

    def forward(self, x):
        x = x.view(-1, self.size[0])
        for layer in self.layers:
            if layer is self.layers[-1]:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x

    def all_hidden_neurons(self, x):
        hidden_neurons = []
        x = x.view(self.size[0])
        for layer in self.layers[:-1]:
            if layer is self.layers[0]:
                x = layer(x)
            else:
                x = layer(F.relu(x))
            hidden_neurons.append(x)
        return torch.cat(hidden_neurons, dim=-1)

    def activation_pattern(self, x):
        x_activation_pattern = self.all_hidden_neurons(x) > 0
        return [entry.item() for entry in x_activation_pattern]

model = MLPNet([600, 256,20])
# model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.5,lr=0.001)
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        # data, label = data.cuda(), label.cuda()
        optimizer.zero_grad() #梯度变0，不让梯度进行累加
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

def val(epoch):
    model.eval() #验证
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():  # 不进行梯度计算
        for data, label in test_loader:
            # data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)   # 损失不回传
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
start = time.time()
for epoch in range(1, epochs+1):
    train(epoch)
    val(epoch)
cost_time = time.time() - start
print('cost time:', cost_time)
save_path = "./PurchaseModel_class.pt"
torch.save(model.state_dict(), save_path)