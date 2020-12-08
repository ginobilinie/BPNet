# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import config
from seg_opr.seg_oprs import ConvBnRelu
#from aspp import build_aspp
from ppm import PPM
from apnb import APNB
from afnb import AFNB
import numpy as np

'''
Dong Nie, Nov. 2019
dong.nie@alibaba-inc.com
'''


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out

class InterFeatureFusion(nn.Module):
    def __init__(self, source_in_planes, target_in_planes, bn_eps=1e-5,
                 bn_momentum=0.1, inplace=True, norm_layer=nn.BatchNorm2d):
        super(InterFeatureFusion, self).__init__()
        # self.convSource = nn.Sequential(
        #     nn.Conv2d(source_in_planes, target_in_planes, 1, bias=False)
        #     norm_layer(target_in_planes, eps=bn_eps, momentum=bn_momentum)
        #     )
        # self.convTarget = nn.Sequential(
        #     nn.Conv2d(target_in_planes, target_in_planes, 1, bias=False)
        #     norm_layer(target_in_planes, eps=bn_eps, momentum=bn_momentum)
        #     )
        self.convSource1 = ConvBnRelu(source_in_planes, target_in_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        self.convSource2 = ConvBnRelu(target_in_planes, target_in_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        self.convTarget = ConvBnRelu(target_in_planes, target_in_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        # self.relu_inplace = nn.ReLU(inplace=True)

    def forward(self, source_feature, target_feature):
        source_feature = F.interpolate(source_feature, size=target_feature.shape[2::], mode='bilinear',
                                       align_corners=True)
        source_feature = self.convSource1(source_feature)
        # target_feature = self.convTarget(target_feature)
        out = source_feature + target_feature + self.convSource2(source_feature * target_feature)
        # out = self.relu_inplace(out)
        out = self.convTarget(out)
        return out

class InterFeatureDownsample(nn.Module):
    def __init__(self, inplanes, planes, scale=1, bn_eps=1e-5,
                 bn_momentum=0.1, inplace=True, norm_layer=nn.BatchNorm2d):
        super(InterFeatureDownsample, self).__init__()
        # self.downsamplelayers = []
        # for i in range(channel):
        #     self.downsamplelayers.append(ConvBnRelu(inplanes, inplanes*2, 3, 1, 1,
        #                       has_bn=True, norm_layer=norm_layer,
        #                       has_relu=True, inplace=inplace, has_bias=False))
        #     inplanes = inplanes*2
        # self.downsamplelayers = nn.Sequential(*self.downsamplelayers)
        self.scale = scale
        self.downsamplelayers = ConvBnRelu(inplanes, planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        self.convTimes = ConvBnRelu(planes, planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        # self.relu_inplace = nn.ReLU(inplace=True)
    def forward(self, x, fm):
        x = F.interpolate(x, size=fm.shape[2::],
                          mode='bilinear', align_corners=True)
        x = self.downsamplelayers(x)
        out = x + fm + self.convTimes(x * fm)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, bins=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True, alpha=1):

        stem_width = int(np.rint(stem_width * alpha))
        self.inplanes = stem_width * 2 if deep_stem else int(np.rint(64 * alpha))
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, int(np.rint(64 * alpha)), kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else int(np.rint(64 * alpha)), eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, int(np.rint(64 * alpha)), layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, int(np.rint(128 * alpha)), layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, int(np.rint(256 * alpha)), layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, int(np.rint(512 * alpha)), layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer5 = self._make_layer(block, norm_layer, int(np.rint(1024 * alpha)), layers[4],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        # self.aspp = build_aspp(int(np.rint(512 * alpha)), int(np.rint(512 * alpha)), 8, norm_layer)
        # self.ppm2 = PPM(int(np.rint(128 * alpha)), int(int(np.rint(128 * alpha)) / len(bins)), bins, norm_layer)
        # self.ppm3 = PPM(int(np.rint(256 * alpha)), int(int(np.rint(256 * alpha)) / len(bins)), bins, norm_layer)
        # self.ppm4 = PPM(int(np.rint(512 * alpha)), int(int(np.rint(512 * alpha))/len(bins)), bins, norm_layer)
        self.ppm = PPM(int(np.rint(1024 * alpha)), int(int(np.rint(1024 * alpha)) / len(bins)), bins, norm_layer) #placed on the smallest resolution
        self.fuse2_1 = InterFeatureFusion(int(np.rint(128 * alpha)), int(np.rint(64 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse3_1 = InterFeatureFusion(int(np.rint(256 * alpha)), int(np.rint(128 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse3_2 = InterFeatureFusion(int(np.rint(128 * alpha)), int(np.rint(64 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse4_1 = InterFeatureFusion(int(np.rint(512 * alpha)), int(np.rint(256 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse4_2 = InterFeatureFusion(int(np.rint(256 * alpha)), int(np.rint(128 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse4_3 = InterFeatureFusion(int(np.rint(128 * alpha)), int(np.rint(64 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse5_1 = InterFeatureFusion(int(np.rint(1024 * alpha)), int(np.rint(512 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse5_2 = InterFeatureFusion(int(np.rint(512 * alpha)), int(np.rint(256 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse5_3 = InterFeatureFusion(int(np.rint(256 * alpha)), int(np.rint(128 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse5_4 = InterFeatureFusion(int(np.rint(128 * alpha)), int(np.rint(64 * alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)

        self.down2 = InterFeatureDownsample(int(np.rint(64 * alpha)), int(np.rint(128 * alpha)), scale=1/2, bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.down3 = InterFeatureDownsample(int(np.rint(64 * alpha)), int(np.rint(256 * alpha)), scale=1/4, bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.down4 = InterFeatureDownsample(int(np.rint(64 * alpha)), int(np.rint(512 * alpha)), scale=1/8, bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)


        self.conv_low = ConvBnRelu(int(np.rint(64 * alpha)), int(np.rint(512 * alpha)), 4, 4, 0,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        self.conv_high = ConvBnRelu(int(np.rint(64 * alpha)), int(np.rint(512 * alpha)), 4, 4, 0,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        # low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout
        # self.fusion = AFNB(int(np.rint(1024 * alpha)), 2048, 2048, int(np.rint(256 * alpha)), int(np.rint(256 * alpha)), dropout=0.05, sizes=([1]), norm_layer=norm_layer)
        self.fusion = AFNB(int(np.rint(512 * alpha)), int(np.rint(512 * alpha)), int(np.rint(512 * alpha)), int(np.rint(128 * alpha)), int(np.rint(128 * alpha)), dropout=0.05, sizes=([1]), norm_layer=norm_layer)
        # extra added layers
        self.context = nn.Sequential(
            # nn.Conv2d(2048, int(np.rint(512 * alpha)), kernel_size=3, stride=1, padding=1),
            # ModuleHelper.BNReLU(int(np.rint(512 * alpha)), norm_type=self.configer.get('network', 'norm_type')),
            ConvBnRelu(int(np.rint(512 * alpha)), int(np.rint(512 * alpha)), 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False),
            # APNB(in_channels=int(np.rint(512 * alpha)), out_channels=int(np.rint(512 * alpha)), key_channels=int(np.rint(256 * alpha)), value_channels=int(np.rint(256 * alpha)),
            #      dropout=0.05, sizes=([1]), norm_type=self.configer.get('network', 'norm_type'))
            APNB(in_channels=int(np.rint(512 * alpha)), out_channels=int(np.rint(512 * alpha)), key_channels=int(np.rint(128 * alpha)), value_channels=int(np.rint(128 * alpha)),
                         dropout=0.05, sizes=([1]), norm_layer=norm_layer)
        )
        self.depthwise_conv = nn.Conv2d(int(np.rint(512 * alpha)), int(np.rint(512 * alpha)), kernel_size=3, stride=1, padding=1, groups=int(np.rint(512 * alpha)), bias=True)
        self.head = nn.Sequential(ConvBnRelu(int(np.rint(512 * alpha))+int(np.rint(64 * alpha)), int(np.rint(64 * alpha)), 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False),
                                  nn.Conv2d(int(np.rint(64 * alpha)), int(np.rint(64 * alpha)), kernel_size=1, stride=1, padding=0, bias=True),
                                  nn.Dropout2d(0.2)  # added to prevent overfitting
                                 )

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        fm2 = self.layer2(x)
        x = self.fuse2_1(fm2, x)
        blocks.append(x)
        fm3 = self.layer3(self.down2(x, fm2))
        fm2 = self.fuse3_1(fm3, fm2)
        x = self.fuse3_2(fm2, x)
        blocks.append(x)

        fm4 = self.layer4(self.down3(x, fm3))
        #fm4 = self.ppm(fm4) # we can embed into somewhere else
        fm3 = self.fuse4_1(fm4, fm3)
        fm2 = self.fuse4_2(fm3, fm2)
        x = self.fuse4_3(fm2, x)
        blocks.append(x)

        fm5 = self.layer5(self.down4(x,fm4))
        fm4 = self.fuse5_1(self.ppm(fm5), fm4) #use ppm here
        fm3 = self.fuse5_2(fm4, fm3)
        fm2 = self.fuse5_3(fm3, fm2)
        x = self.fuse5_4(fm2, x)
        blocks.append(x)
        #asymmetric non-local neural network

        x = self.fusion(self.conv_low(blocks[-2]), self.conv_high(blocks[-1])) # the channel size is too small
        x = self.context(x)
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        # #print(x.shape, 'xxxx',blocks[-1].shape)
        # x = self.head(torch.cat((blocks[-1],x),dim=1))
        # blocks.append(x)

        #now comes local distribution alogrithm on small resolution feature maps
        map = F.sigmoid(self.depthwise_conv(x)) #we obtain int(np.rint(512 * alpha))-channel feature map
        # map = F.sigmoid(map)
        x = map*x + x
        x = F.interpolate(x, size=blocks[-1].shape[2::], mode='bilinear', align_corners=True)
        x = self.head(torch.cat((x, blocks[-1]),dim=1)) # concat feature before self-attention
        blocks.append(x)



        return blocks

def resnet18(pretrained_model=None, num_classes=19, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2, 2], num_classes=num_classes, **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model

class FPNet(nn.Module):
    def __init__(self, out_planes, is_training, 
                 criterion, ohem_criterion,
                 dice_criterion, dice_weight=1,
                 inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d, alpha=1):
        super(FPNet, self).__init__()
        self.is_training = is_training
        self.backbone = resnet18(pretrained_model, num_classes=out_planes, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=True, stem_width=32, alpha=alpha)

        heads = [FPNetHead(int(np.rint(64 * alpha)), out_planes, 4,
                             False, norm_layer, alpha=alpha),
                 FPNetHead(int(np.rint(64 * alpha)), out_planes, 4,
                             False, norm_layer, alpha=alpha),
                 FPNetHead(int(np.rint(64 * alpha)), out_planes, 4,
                           False, norm_layer, alpha=alpha),
                 FPNetHead(int(np.rint(64 * alpha)), out_planes, 4,
                             False, norm_layer, alpha=alpha),
                 FPNetHead(int(np.rint(64 * alpha)), out_planes, 4,
                           False, norm_layer, alpha=alpha)
                 ]

        self.heads = nn.ModuleList(heads)

        self.business_layer = []
        self.business_layer.append(self.backbone)
        self.business_layer.append(self.heads)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion
            self.dice_criterion = dice_criterion
            self.dice_weight = dice_weight

    def forward(self, data, label=None):
        blocks = self.backbone(data)

        if self.is_training:
            pred0 = self.heads[0](blocks[0])
            pred1 = self.heads[1](blocks[1])
            pred2 = self.heads[2](blocks[2])
            pred3 = self.heads[3](blocks[3])
            pred4 = self.heads[4](blocks[4])
            aux_loss0 = self.ohem_criterion(pred0, label)
            aux_loss1 = self.ohem_criterion(pred1, label)
            aux_loss2 = self.ohem_criterion(pred2, label)
            aux_loss3 = self.ohem_criterion(pred3, label)
            main_loss = self.ohem_criterion(pred4, label)

            loss_ce = main_loss + 0.4 * aux_loss0 + 0.4 * aux_loss1 + 0.4 * aux_loss2 + 0.7 * aux_loss3

            loss_dice = torch.FloatTensor(0)
            if self.dice_criterion is not None:
                dice_loss0 = self.dice_criterion(pred0, label)
                dice_loss1 = self.dice_criterion(pred1, label)
                dice_loss2 = self.dice_criterion(pred2, label)
                dice_loss3 = self.dice_criterion(pred3, label)
                dice_loss_main = self.dice_criterion(pred4, label)

                loss_dice = (dice_loss0 * 0.4 + dice_loss1 * 0.4 + dice_loss2 * 0.4 + dice_loss3 * 0.7 + dice_loss_main) * self.dice_weight

            return loss_ce+loss_dice, loss_ce, loss_dice

        return F.log_softmax(self.heads[-1](blocks[-1]), dim=1)

class FPNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d, alpha=1):
        super(FPNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, int(np.rint(256 * alpha)), 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, int(np.rint(64 * alpha)), 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(int(np.rint(256 * alpha)), out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(int(np.rint(64 * alpha)), out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, size=[config.image_height, config.image_width],
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    import time
    model = FPNet(21, None, None, None)
    alpha = config.alpha
    img = torch.randn(2, 3, int(np.rint(256 * alpha)), int(np.rint(int(np.rint(512 * alpha)) * alpha)))
    start = time.time()
    for i in range(10):
      output = model(img)
    print('time:', time.time()-start)
    print(output.shape)
