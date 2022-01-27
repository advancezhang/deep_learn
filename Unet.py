import paddle
import paddle.nn as nn

class DoubleConv(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2D(out_ch),
            nn.ReLU(),
            nn.Conv2D(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2D(out_ch),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Layer):
    def __init__(self,num_classes):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2D(2,2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2D(2,2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2D(2,2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2D(2,2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.Conv2DTranspose(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.Conv2DTranspose(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.Conv2DTranspose(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.Conv2DTranspose(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2D(64, num_classes, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = paddle.concat([up_6, c4], axis=1)
        c6=self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = paddle.concat([up_7, c3], axis=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = paddle.concat([up_8, c2], axis=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = paddle.concat([up_9, c1], axis=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10
      
model=Unet(num_classes=2)
paddle.summary(model,(10,3,32,32))
x = paddle.randn((10, 3, 32, 32))
# 测试模型前向计算
y = model(x)
# 打印输出维度
print(y.shape)
