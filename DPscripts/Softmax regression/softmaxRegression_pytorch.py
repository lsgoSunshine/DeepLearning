import torch
import torchvision
# 常用的图片转换，如裁剪旋转等。
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import dlzh.d2lzh_pytorch as d2l
from dlzh import timer

'''
2. 实现softmax运算
'''


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


'''
进行加权运算
'''


def net(X):
    return softmax(torch.mm(X.view(-1, num_inputs), W) + b)


'''
交叉熵损失函数
'''


# gather函数针对dim和后面给出index 张量进行聚类，

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


'''计算精确度'''


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 使用 Fashion-MNIST
if __name__ == '__main__':
    # 获取训练数据集和测试数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, 4)
    run_time = timer.Timer()
    '''
    1. 初始化模型参数：已知每个样本都是高和宽均为28个像素的的图像，因此模型输入是28*28=784，
    又因为图像的标签有10个类别，因此输出个数也是10
    '''
    num_inputs = 784
    num_outputs = 10
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    print(W, b)
    # #测试
    # y_hat = torch.tensor([[.1,.3,.6],[.3,.2,.5]])
    # y=torch.LongTensor([0,2])
    # print(y.view(-1,1))
    # print(y_hat.gather(1,y.view(-1,1)))
    #
    # print(accuracy(y_hat,y))
    # print((y_hat.argmax(dim=1)==y).float().mean().item())

    num_epochs, lr = 5, .1
    loss = cross_entropy
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, [W, b], lr)

    # 预测
    X, y = iter(test_iter).next()
    print(X, y)
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    print(net(X))
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])
