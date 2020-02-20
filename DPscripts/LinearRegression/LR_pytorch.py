import torch
import time
import numpy as np
# 构造数据集的包
import torch.utils.data as Data
# neural network 包
from torch import nn
# 初始化模型需要用的包
from torch.nn import init
# 提供优化算法的包
import torch.optim as optim

num_inputs = 2
num_examples = 1000
''' true_w和true_b是真实的权重w和置偏值b'''
true_w = torch.tensor([2, -3.4])
true_b = 4.2
'''
步骤一：     首先构造训练数据集X和Y，

X随机生成以0为中心方差是1的1000*2的数据集
Y是使用生成的X带入真实的w,b计算得到
为什么是1000*2呢?
说白了就是1000行2列的矩阵，因为你这个矩阵一定是训练数据矩阵X，它会去乘以你的初始权重矩阵w，
而初始权重矩阵显而易见是一个2*1的矩阵，因此1000*2*2*1才能得到1000个标签（labels）
所以碰到不清楚矩阵的形状（shape）的时候可以想一想，需要得到什么结果，这个矩阵该怎么进行运算。
'''
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
# labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
'''
矩阵相乘有torch.mm和torch.matmul两个函数。其中前一个是针对二维矩阵，后一个是高维。当torch.mm用于大于二维时将报错。
上面那个矩阵乘法得到的结果和下面是一致的
'''
labels = torch.matmul(features, torch.t(true_w)) + true_b

'''
步骤二：读取数据
 
Batch_size 的选择，首先决定的是下降的方向。如果数据集比较小，完全可以采用全数据集 （ Full Batch Learning ）的形式，这样做至少有 2 个好处：
其一，由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。
其二，由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 
Full Batch Learning 可以使用 Rprop 只基于梯度符号并且针对性单独更新各权值。
对于更大的数据集，以上 2 个好处又变成了 2 个坏处：
其一，随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。
其二，以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。这才有了后来 RMSProp 的妥协方案。
'''
batch_size = 10
# 将训练集和标签进行组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量,这里的shuffle参数是用来打乱数据集的
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# 可以打印出第一个小批量的数据样本看一下
for X, y in data_iter:
    print(X, y)
    break

'''
步骤三 : 定义线性回归模型,搭建神经网络

导入torch.nn模块，nn是neural network的缩写。该模块中定义了大量的神经网络的层，nn就是利用autograd来定义模型的
nn的核心数据结构是Module，它是一个抽象的概念，既可以表示神经网络的某个层，也可以表示一个包含很多层的神经网络。
实际使用中，最常见的就是继承nn.Module
'''


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


# 这里我们就可以看到我们已经搭建了一个单层的线性回归网络，有俩输入，一个输出
# net = LinearNet(num_inputs)
# print(net)
'''
还可以使用nn.Sequential来搭建网络，Sequential是一个有序的容器，网络层按照他的顺序依次被添加到计算图中
'''
# 写法1
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
)
# # 写法2
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......
# # 写法3
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, 1))
#     # ......
# ]))
print(net)
print(net[0])
'''
可以通过net.parameters()来查看模型所有可以学习参数，此函数返回一个生成器
'''
for param in net.parameters():
    print(param)

'''
需要注意的是，torch.nn仅支持一个batch的样本输入，而不支持单个样本输入，如果有单个样本，可以使用input.unsqueeze(0)来添加一维
'''

'''
步骤四 ： 初始化模型参数.
偏差初始化为0
'''
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

'''
步骤五：定义损失函数
使用pytorch提供的均方误差作为模型的损失函数
'''
loss = nn.MSELoss()

'''
步骤六：定义优化算法(损失函数)
torch.optim提供很多常用的优化算法，指定学习率为0.03
'''
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

'''
步骤七 ：训练模型。通过optim实例的step函数来迭代模型参数，按照小批量随机梯度下降的定义，在step函数中指明批量大小，再对批量样本求平均
'''
num_enpochs = 3
for epoch in range(1, num_enpochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))# 这里对y进行转置
        optimizer.zero_grad()  # set grad zero
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

'''
步骤八：比较真实的参数和模型参数
'''
print(net[0].weight, true_w)
print(net[0].bias, true_b)
