import torch
from torch import nn
from torchkeras import summary

print("torch version: %s"%torch.__version__)

class ConvBN(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBN,self).__init__()
        self.conv=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU()

    def forward(self,x):
        y=self.conv(x)
        y=self.bn(y)
        y=self.act(y)
        return y

class CNN(nn.Module):
    def __init__(self,stage_channels, num_classes):
        super(CNN, self).__init__()
        layers = nn.ModuleList()
        for i, o in zip(stage_channels, stage_channels[1:]):
            print(i, o)
            layer = ConvBN(in_channels=i, out_channels=o,
                           kernel_size=3, stride=1, padding=1)
            layers.append(layer)
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_channels=stage_channels[-1], out_channels=num_classes,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.head(y)
        return y
model = CNN(stage_channels=[3, 8, 16, 32, 16, 8], num_classes=2)
# 生产一个随机数据
print(model)
x = torch.randn((1, 3, 32, 32))
print(x)
# 测试模型前向计算
y = model(x)
print(y)
# 打印输出维度
print(y.shape)
summary(model,(3, 32, 32))
