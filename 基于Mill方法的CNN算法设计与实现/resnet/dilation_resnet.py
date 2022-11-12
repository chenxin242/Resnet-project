import paddle
from paddle import nn
from paddle.vision.models.resnet import BasicBlock

#加入空洞卷积的resnet
class ResNet(nn.Layer):
    def __init__(self, block, depth, num_classes=10):
        super(ResNet, self).__init__()
        layer_cfg = {
            20: [3, 3, 3],
            32: [5, 5, 5],
            44: [7, 7, 7],
            56: [9, 9, 9],
            110:[18, 18, 18],
            1202:[200, 200, 200],
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 16
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            #定义空洞卷积的大小
            dilation=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        #空洞卷积的大小赋值
        previous_dilation = self.dilation
        #处理维度不一样shortcut为虚线情况
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        #传入卷积
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 16,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 3, 32, 32], name='x')])
   #定义前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        #paddle的flatten将其战平为一维
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x

    
def myresnet20():
    return ResNet(BasicBlock, 20)

def myresnet32():
    return ResNet(BasicBlock, 32)

def myresnet44():
    return ResNet(BasicBlock, 44)

def myresnet56():
    return ResNet(BasicBlock, 56)

def myresnet110():
    return ResNet(BasicBlock, 110)

def myresnet1202():
    return ResNet(BasicBlock, 1202)



