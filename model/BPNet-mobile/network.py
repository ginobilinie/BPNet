# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from config import config
from blocks import ConvBnRelu
# from aspp import build_aspp
from ppm import PPM
from apnb import APNB
from afnb import AFNB
from blocks import ConvBnRelu_DW, ConvBn_DW

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
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        # self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)

        self.convbnrelu_dw1 = ConvBnRelu_DW(inplanes, planes, ksize=3, stride=stride, pad=1)

        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        # self.stride = stride
        self.inplace = inplace

        self.convbn_dw2 = ConvBn_DW(planes, planes, ksize=3, stride=1, pad=1)


    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.convbnrelu_dw1(x)

        # out = self.conv2(out)
        # out = self.bn2(out)

        out = self.convbn_dw2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        #print('out.shape: ',out.shape,' residual.shape: ',residual.shape)
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
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)

        # self.convbnrelu_dw1 = ConvBnRelu_DW(inplanes, planes, ksize=1, stride=1, pad=0)

        self.convbnrelu_dw2 = ConvBnRelu_DW(planes, planes, ksize=3, stride=stride, pad=1)

        # self.convbn_dw3 = ConvBn_DW(planes, planes * self.expansion, ksize=1, stride=1, pad=0)

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

        # out = self.convbnrelu_dw1(x)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        out = self.convbnrelu_dw2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.convbn_dw3(out)

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

        self.convSource1 = ConvBnRelu_DW(source_in_planes, target_in_planes, 3, 1, 1, norm_layer=norm_layer)

        self.convSource2 = ConvBnRelu_DW(target_in_planes, target_in_planes, 3, 1, 1, norm_layer=norm_layer)

        self.convTarget = ConvBnRelu_DW(target_in_planes, target_in_planes, 3, 1, 1, norm_layer=norm_layer)

    def forward(self, source_feature, target_feature):
        #source_feature = F.interpolate(source_feature, scale_factor=2, mode='bilinear', align_corners=True)
        source_feature = F.interpolate(source_feature, [target_feature.shape[-2], target_feature.shape[-1]], mode='bilinear', align_corners=True)
        source_feature = self.convSource1(source_feature)
        out = source_feature + target_feature + self.convSource2(source_feature * target_feature)
        out = self.convTarget(out)
        return out

class InterFeatureDownsample(nn.Module):
    def __init__(self, inplanes, planes, scale=1, bn_eps=1e-5, 
                 bn_momentum=0.1, inplace=True, norm_layer=nn.BatchNorm2d):
        super(InterFeatureDownsample, self).__init__()

        self.scale = scale
        self.downsamplelayers = ConvBnRelu_DW(inplanes, planes, 3, 1, 1, norm_layer=norm_layer)

        self.convTimes = ConvBnRelu_DW(planes, planes, 3, 1, 1, norm_layer=norm_layer)

        # self.relu_inplace = nn.ReLU(inplace=True)
    def forward(self, x, fm):
        #x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=[fm.shape[-2],fm.shape[-1]], mode='bilinear', align_corners=True)
        x = self.downsamplelayers(x)
        #print('x.shape: ',x.shape,' fm.shape:',fm.shape)
        out = x + fm + self.convTimes(x * fm)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, bins=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True, alpha=1):
        stem_width = int(np.rint(stem_width * alpha))
        self.inplanes = stem_width * 2  if deep_stem else int(np.rint(64 * alpha))
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=4,
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
            self.conv1 = nn.Conv2d(3, int(np.rint(64*alpha)), kernel_size=3, stride=2, padding=4,
                                   bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else int(np.rint(64*alpha)), eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, int(np.rint(64*alpha)), layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, int(np.rint(128*alpha)), layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, int(np.rint(256*alpha)), layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, int(np.rint(512*alpha)), layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.ppm = PPM(int(np.rint(512*alpha)), int(np.rint(512*alpha) // len(bins)), bins, norm_layer)
        self.fuse2_1 = InterFeatureFusion(int(np.rint(128*alpha)), int(np.rint(64*alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse3_1 = InterFeatureFusion(int(np.rint(256*alpha)), int(np.rint(128*alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse3_2 = InterFeatureFusion(int(np.rint(128*alpha)), int(np.rint(64*alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse4_1 = InterFeatureFusion(int(np.rint(512*alpha)), int(np.rint(256*alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse4_2 = InterFeatureFusion(int(np.rint(256*alpha)), int(np.rint(128*alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.fuse4_3 = InterFeatureFusion(int(np.rint(128*alpha)), int(np.rint(64*alpha)), bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)

        self.down2 = InterFeatureDownsample(int(np.rint(64*alpha)), int(np.rint(128*alpha)), scale=0.5, bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.down3 = InterFeatureDownsample(int(np.rint(64*alpha)), int(np.rint(256*alpha)), scale=0.25, bn_eps=bn_eps,
                 bn_momentum=bn_momentum, norm_layer=norm_layer)


        self.conv_low = ConvBnRelu_DW(int(np.rint(64*alpha)), int(np.rint(512*alpha)), 4, 4, 1, norm_layer=norm_layer)
        self.conv_high = ConvBnRelu_DW(int(np.rint(64*alpha)), int(np.rint(512*alpha)), 4, 4, 1, norm_layer=norm_layer)


        self.fusion = AFNB(int(np.rint(512*alpha)), int(np.rint(512*alpha)), int(np.rint(512*alpha)), int(np.rint(128*alpha)), int(np.rint(128*alpha)), dropout=0.05, sizes=([1]), norm_layer=norm_layer)
        self.context = nn.Sequential(
            ConvBnRelu_DW(int(np.rint(512*alpha)), int(np.rint(512*alpha)), 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False),
            APNB(in_channels=int(np.rint(512*alpha)), out_channels=int(np.rint(512*alpha)), key_channels=int(np.rint(128*alpha)), value_channels=int(np.rint(128*alpha)),
                         dropout=0.05, sizes=([1]), norm_layer=norm_layer)
        )
        self.head = nn.Sequential(ConvBnRelu_DW(int(np.rint((512+64)*alpha)), int(np.rint(64*alpha)), 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False),
                                  nn.Conv2d(int(np.rint(64*alpha)), int(np.rint(64*alpha)), kernel_size=1, stride=1, padding=0, bias=True),
                                  nn.Dropout2d(0.25)  # added to prevent overfitting, 0.1->0.2
                                  )

        self.depthwise_conv = nn.Conv2d(int(np.rint(512*alpha)), int(np.rint(512*alpha)),kernel_size=3,stride=1,padding=1,groups=int(np.rint(512*alpha)), bias=True)


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
        #x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        fm2 = self.layer2(x)
        x = self.fuse2_1(fm2, x)
        blocks.append(x)
        fm3 = self.layer3(self.down2(x, fm2))
        fm2 = self.fuse3_1(fm3, fm2)
        x = self.fuse3_2(fm2, x)
        blocks.append(x)
        # xxx = self.down3(x, fm3)
        fm4 = self.layer4(self.down3(x, fm3))

        fm4 = self.ppm(fm4) # we can embed into somewhere

        fm3 = self.fuse4_1(fm4, fm3)
        fm2 = self.fuse4_2(fm3, fm2)
        x = self.fuse4_3(fm2, x)
        blocks.append(x)
        
        #print('before fusion: x.shape: ',x.shape, 'blocks[-2].shape: ',blocks[-2].shape)
        x = self.fusion(self.conv_low(blocks[-2]), self.conv_high(blocks[-1]))
        #print('after fusion: x.shape: ',x.shape)
        x = self.context(x) # output channel is 512
        #print('after context: x.shape: ',x.shape)
        map = F.sigmoid(self.depthwise_conv(x))
        x = map*x + x
        x = F.interpolate(x, size=(blocks[-1].shape[-2],blocks[-1].shape[-1]), mode='bilinear', align_corners=True)
        #print('x.shape: ',x.shape,' blocks[-1].shape: ',blocks[-1].shape)
        x = self.head(torch.cat((x, blocks[-1]),dim=1)) # concat feature before self-attention
        blocks.append(x)

        return blocks

def resnet18(pretrained_model=None, num_classes=19, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model

class FPNet(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, ohem_criterion, dice_criterion,
                 dice_weight=1,
                 inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d, alpha=1):
        super(FPNet, self).__init__()
        self.is_training = is_training
        self.backbone = resnet18(pretrained_model, num_classes=out_planes, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=True, stem_width=32, alpha=alpha)

        heads = [FPNetHead(int(np.rint(64*alpha)), out_planes, 4,
                             False, norm_layer, alpha=alpha),
                 FPNetHead(int(np.rint(64*alpha)), out_planes, 4,
                             False, norm_layer, alpha=alpha),
                 FPNetHead(int(np.rint(64*alpha)), out_planes, 4,
                           False, norm_layer,alpha=alpha),
                 FPNetHead(int(np.rint(64*alpha)), out_planes, 4,
                             False, norm_layer,alpha=alpha)]

        self.heads = nn.ModuleList(heads)


        self.business_layer = []
        self.business_layer.append(self.backbone)
        self.business_layer.append(self.heads)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion
            self.dice_criterion = dice_criterion
            self.dice_weight = dice_weight


    def forward(self, data, label=None, instance_label=None):
        blocks = self.backbone(data)

        if self.is_training:
            pred0 = self.heads[0](blocks[0])
            pred1 = self.heads[1](blocks[1])
            pred2 = self.heads[2](blocks[2])
            pred3 = self.heads[3](blocks[3])

            aux_loss0 = self.ohem_criterion(pred0, label)
            aux_loss1 = self.ohem_criterion(pred1, label)
            aux_loss2 = self.ohem_criterion(pred2, label)
            main_loss = self.ohem_criterion(pred3, label)

            loss_ce = main_loss + 0.4 * aux_loss0 + 0.4 * aux_loss1 + 0.7 * aux_loss2
            loss_dice = torch.FloatTensor(0)
            if self.dice_criterion is not None:
                dice_loss0 = self.dice_criterion(pred0, label)
                dice_loss1 = self.dice_criterion(pred1, label)
                dice_loss2 = self.dice_criterion(pred2, label)
                dice_loss_main = self.dice_criterion(pred3, label)

                loss_dice = (dice_loss0*0.4 + dice_loss1*0.4 + dice_loss2*0.7 + dice_loss_main)*self.dice_weight

            return loss_dice+loss_ce, loss_ce, loss_dice

        #return F.log_softmax(self.heads[-1](blocks[-1]), dim=1)
        return F.softmax(self.heads[-1](blocks[-1]), dim=1)

class FPNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d, alpha=1):
        super(FPNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, int(np.rint(256*alpha)), 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, int(np.rint(64*alpha)), 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(int(np.rint(256*alpha)), out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(int(np.rint(64*alpha)), out_planes, kernel_size=1,
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
    img = torch.randn(2, 3, 256, 512)
    start = time.time()
    for i in range(10):
      output = model(img)
    print('time:', time.time()-start)
    print(output.shape)
