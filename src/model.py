import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from config import *
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet12(nn.Module):
    def __init__(self, num_blocks_list=[2, 2, 4, 4], num_classes=NUM_CLASSES):
        super().__init__()
        # 12 blocks: 2+2+4+4 = 12
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # Each layer
        self.layer1 = self._make_layer(32, num_blocks_list[0])
        self.layer2 = self._make_layer(64, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks_list[3], stride=2)
        # 作为 FasterRCNN backbone，不需要分类头
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)

        # 关键属性：输出通道数
        self.out_channels = 256

    def _make_layer(self, planes, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # FasterRCNN 要求返回字典
        return {"0": x}

# 实例化 ResNet12
backbone = ResNet12(num_blocks_list=[2, 2, 4, 4])

# Anchor生成器和ROI Pooler（保持默认即可）
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# 构建 FasterRCNN 模型
model = FasterRCNN(
    backbone,
    num_classes=NUM_CLASSES,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

# 现在你可以训练或推理了
# 输入图片需转成Tensor，模型输入同标准FasterRCNN