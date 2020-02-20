import torch
import math
import torchvision
# 常用的图片转换，如裁剪旋转等。
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
# sys.path.append("./dlzh")
# import d2lzh_pytorch as d2l

a= torch.tensor([1,2,3,4,5,6])
print(a.view(-1,1))

def softmax(a):
    sum=0
    for i in a :
        sum+=math.exp(i)
    return [math.exp(i)/sum for i in a ]


print(softmax([100,101,102]))
print(softmax([10.0, 10.1, 10.2]))
print(softmax([-100, -101, -102]))
print(softmax([-2 -1, 0.0001]))
print(softmax([1000, 1010, 1020]))