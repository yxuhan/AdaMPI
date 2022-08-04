# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


'''
This code is borrowed heavily from MINE: https://github.com/vincentfung13/MINE
'''


import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入为RGBD
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnet_multiimage_input(num_layers, num_input_images)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.img_mean = self.img_mean.view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.img_std = self.img_std.view(1, 3, 1, 1)

    def forward(self, input_image, input_depth):
        # normalize before going into network
        ref_images_normalized = (input_image - self.img_mean.to(input_image)) / self.img_std.to(input_image)

        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = torch.cat([ref_images_normalized, input_depth], dim=1) 
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        conv1_out = self.encoder.relu(x)  # [bs,64,h//2,w//2]
        block1_out = self.encoder.layer1(self.encoder.maxpool(conv1_out))  # [bs,256,h//4,w//4]
        block2_out = self.encoder.layer2(block1_out)  # [bs,512,h//8,w//8]
        block3_out = self.encoder.layer3(block2_out)  # [bs,1024,h//16,w//16]
        block4_out = self.encoder.layer4(block3_out)  # [bs,2048,h//32,w//32]

        return conv1_out, block1_out, block2_out, block3_out, block4_out
