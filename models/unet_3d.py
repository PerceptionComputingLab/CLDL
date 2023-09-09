import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
import torch.nn as nn
# from src.models import resnet
# from src.miscellaneous.utils import entropy
# from src.models.segmentor_v1 import DenseNetDist
# from models import resnet
from miscellaneous.utils import entropy



class EncodeBasicBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, padding):
        super(EncodeBasicBlock, self).__init__()
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=middle_channels, kernel_size=kernel_size, padding=padding)
        self.bn0= nn.BatchNorm3d(middle_channels)
        self.conv1 = nn.Conv3d(in_channels=middle_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class DecodeBasicBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, padding):
        super(DecodeBasicBlock, self).__init__()
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=middle_channels, kernel_size=kernel_size, padding=padding)
        self.bn0= nn.BatchNorm3d(middle_channels)
        self.conv1 = nn.Conv3d(in_channels=middle_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class Unet3D(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, softmax=False):
        super(Unet3D, self).__init__()
        self.softmax = softmax
        self.conv1 = EncodeBasicBlock(input_dim, 32, 64, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = EncodeBasicBlock(64, 64, 128, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = EncodeBasicBlock(128, 128, 256, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = EncodeBasicBlock(256, 256, 512, (3, 3, 3), (1, 1, 1))

        self.conv5_u = nn.Upsample(scale_factor=2)
        # self.conv5_0 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = DecodeBasicBlock(512+256, 256, 256, (3, 3, 3), (1, 1, 1))

        self.conv6_u = nn.Upsample(scale_factor=2)
        # self.conv6_0 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = DecodeBasicBlock(256+128, 128, 128, (3, 3, 3), (1, 1, 1))

        self.conv7_u = nn.Upsample(scale_factor=2)
        # self.conv7_0 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv7 = DecodeBasicBlock(128+64, 64, 64, (3, 3, 3), (1, 1, 1))

        self.conv8 = nn.Conv3d(in_channels=64, out_channels=output_dim, kernel_size=(1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        #decode
        # up5 = self.conv5_0(self.conv5_u(conv4))
        up5 = self.conv5_u(conv4)
        up5 = torch.cat([up5, conv3], dim=1)
        conv5 = self.conv5(up5)

        # up6 = self.conv6_0(self.conv6_u(conv5))
        up6 = self.conv6_u(conv5)
        up6 = torch.cat([up6, conv2], dim=1)
        conv6 = self.conv6(up6)

        # up7 = self.conv7_0(self.conv6_u(conv6))
        #TODO Big mistake!!!!
        # up7 = self.conv6_u(conv6)
        up7 = self.conv7_u(conv6)
        up7 = torch.cat([up7, conv1], dim=1)
        conv7 = self.conv7(up7)

        conv8 = self.conv8(conv7)
        if self.softmax:
            output = F.softmax(conv8, dim=1)
        else:
            output = conv8
        return output, pool1


class PixelShuffle3D(nn.Module):
    """
    3D version of PixelShuffle Module
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_depth, in_height, in_width = inputs.size()
        channels //= self.upscale_factor ** 3
        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor
        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class ESPCN3D(nn.Module):
    def __init__(self, scale_factor, input_channel, kernel_size, n_class, softmax=False):
        super(ESPCN3D, self).__init__()
        self.softmax = softmax
        self.conv1 = nn.Conv3d(input_channel, 64, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv3 = nn.Conv3d(32, n_class * (scale_factor ** 3), kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.out = PixelShuffle3D(upscale_factor=scale_factor)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(n_class * (scale_factor ** 3))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x1 = F.relu(self.conv1(x))
        # x2 = F.relu(self.conv2(x1))
        # x3 = F.relu(self.conv3(x2))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        if self.softmax:
            x4 = F.softmax(self.out(x3), dim=1)
        else:
            x4 = self.out(x3)
        return x4


class PBFR(nn.Module):
    '''
    This is the implementation of the Prior-based Feature Representation module
    '''
    def __init__(self, input_channel, kernel_size, out_channel, scale_factor):
        super(PBFR, self).__init__()
        # middle_channel = 30
        middle_channel = out_channel//2
        # middle_channel = input_channel*pow(scale_factor, 3)
        self.conv1 = nn.Conv3d(input_channel, middle_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(middle_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm3d(middle_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, boundary_prior, dist_prior, feature):
        # shape of feature should be N*W*D*C
        shape = list(feature.size())
        x1 = F.relu(self.bn1(self.conv1(dist_prior)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        statistic_enhanced_feature = x2*feature
        boundary_prior = torch.unsqueeze(boundary_prior, dim=1)
        bd_prior_multi_channel = boundary_prior.repeat(1,shape[1],1,1,1)
        boundary_enhanced_feature = bd_prior_multi_channel*feature
        out = feature+statistic_enhanced_feature+boundary_enhanced_feature

        return out


class PBFRNew(nn.Module):
    '''
    This is the implementation of the Prior-based Feature Representation module
    '''
    def __init__(self, input_channel, kernel_size, out_channel, scale_factor):
        super(PBFRNew, self).__init__()
        self.conv = CovDropoutNormNonlin(input_channel, out_channel, kernel_size, (1, 1, 1))

    def forward(self, boundary_prior, dist_prior, feature):
        # shape of feature should be N*W*D*C
        shape = list(feature.size())
        x2 = self.conv(dist_prior)
        statistic_enhanced_feature = x2*feature
        boundary_prior = torch.unsqueeze(boundary_prior, dim=1)
        bd_prior_multi_channel = boundary_prior.repeat(1,shape[1],1,1,1)
        boundary_enhanced_feature = bd_prior_multi_channel*feature
        out = feature+statistic_enhanced_feature+boundary_enhanced_feature

        return out


class CovDropoutNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, drop_rate=0, eps=1e-5):
        super(CovDropoutNormNonlin, self).__init__()
        self.drop_rate = drop_rate
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.dropout0 = nn.Dropout3d(drop_rate, inplace=True)
        self.instnorm0 = nn.InstanceNorm3d(out_channels, eps=eps, affine=True)
        self.lrelu0 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.conv1 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.dropout1 = nn.Dropout3d(drop_rate, inplace=True)
        self.instnorm1 = nn.InstanceNorm3d(out_channels, eps=eps, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        if self.drop_rate>0:
            x = self.dropout0(x)
        x = self.instnorm0(x)
        x = self.lrelu0(x)

        x1 = self.conv1(x)
        if self.drop_rate>0:
            x1 = self.dropout1(x1)
        x1 = self.instnorm1(x1)
        x1 = self.lrelu1(x1)

        # TODO test nan error
        flag = torch.any(torch.isnan(x1))
        if flag:
            print("NaN occur!")

        return x1


class NoNewNet(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, softmax=False):
        super(NoNewNet, self).__init__()
        self.softmax = softmax
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        self.conv5_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv5_0 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))

        self.conv6_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv6_0 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = CovDropoutNormNonlin(240+120, 120, (3, 3, 3), (1, 1, 1))

        self.conv7_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv7_0 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv7 = CovDropoutNormNonlin(120+60, 60, (3, 3, 3), (1, 1, 1))

        self.conv8_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv8 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))

        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        #decode
        # up5 = self.conv5_0(self.conv5_u(conv4))
        up5 = self.conv5_u(conv4_bottel)
        up5 = torch.cat([up5, conv4], dim=1)
        conv5 = self.conv5(up5)

        # up6 = self.conv6_0(self.conv6_u(conv5))
        up6 = self.conv6_u(conv5)
        up6 = torch.cat([up6, conv3], dim=1)
        conv6 = self.conv6(up6)

        # up7 = self.conv7_0(self.conv6_u(conv6))
        #TODO Big mistake!!!!
        # up7 = self.conv6_u(conv6)
        up7 = self.conv7_u(conv6)
        up7 = torch.cat([up7, conv2], dim=1)
        conv7 = self.conv7(up7)

        up8 = self.conv8_u(conv7)
        up8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(up8)
        # output layer
        conv9 = self.conv9(conv8)
        if self.softmax:
            output = F.softmax(conv9, dim=1)
        else:
            output = conv9
        return output, pool1


class NoNewNetDeepSP(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, softmax=False):
        super(NoNewNetDeepSP, self).__init__()
        self.softmax = softmax
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        self.conv5_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv5_0 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))

        self.conv6_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv6_0 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = CovDropoutNormNonlin(240+120, 120, (3, 3, 3), (1, 1, 1))

        self.conv7_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv7_0 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv7 = CovDropoutNormNonlin(120+60, 60, (3, 3, 3), (1, 1, 1))

        self.conv8_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv8 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))

        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))


         # auxiliary prediction 0
        self.aux0_conv = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=1, stride=1)
        self.aux0_deconv1 = Upsample(scale_factor=2, mode="trilinear")
        self.aux0_deconv2 = Upsample(scale_factor=2, mode="trilinear")
        self.aux0_prob = Upsample(scale_factor=2, mode="trilinear")
        # auxiliary prediction 1
        self.aux1_conv = nn.Conv3d(in_channels=120, out_channels=output_dim, kernel_size=1, stride=1)
        self.aux1_deconv1 = Upsample(scale_factor=2, mode="trilinear")
        self.aux1_prob = Upsample(scale_factor=2, mode="trilinear")
        # auxiliary prediction 2
        self.aux2_conv = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=1, stride=1)
        self.aux2_prob = Upsample(scale_factor=2, mode="trilinear")

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        #decode
        # up5 = self.conv5_0(self.conv5_u(conv4))
        up5 = self.conv5_u(conv4_bottel)
        up5 = torch.cat([up5, conv4], dim=1)
        conv5 = self.conv5(up5)

        # up6 = self.conv6_0(self.conv6_u(conv5))
        up6 = self.conv6_u(conv5)
        up6 = torch.cat([up6, conv3], dim=1)
        conv6 = self.conv6(up6)

        # up7 = self.conv7_0(self.conv6_u(conv6))
        #TODO Big mistake!!!!
        # up7 = self.conv6_u(conv6)
        up7 = self.conv7_u(conv6)
        up7 = torch.cat([up7, conv2], dim=1)
        conv7 = self.conv7(up7)

        up8 = self.conv8_u(conv7)
        up8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(up8)
        # output layer
        conv9 = self.conv9(conv8)
        if self.softmax:
            output = F.softmax(conv9, dim=1)
        else:
            output = conv9

        # auxilary output 
        _aux0_conv = self.aux0_conv(conv5)
        _aux0_deconv1 = self.aux0_deconv1(_aux0_conv)
        _aux0_deconv2 = self.aux0_deconv2(_aux0_deconv1)
        _aux0_prob = self.aux0_prob(_aux0_deconv2)

        _aux1_conv = self.aux1_conv(conv6)
        _aux1_deconv1 = self.aux1_deconv1(_aux1_conv)
        _aux1_prob = self.aux1_prob(_aux1_deconv1)

        _aux2_conv = self.aux2_conv(conv7)
        _aux2_prob = self.aux2_prob(_aux2_conv)

        return output, pool1, _aux0_prob, _aux1_prob, _aux2_prob


class NoNewNetTrans(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, softmax=False):
        super(NoNewNetTrans, self).__init__()
        self.softmax = softmax
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        self.conv5_u = nn.ConvTranspose3d(480, 480, kernel_size=(2,2,2), stride=(2,2,2))
        # self.conv5_0 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))

        self.conv6_u = nn.ConvTranspose3d(240, 240, kernel_size=(2,2,2), stride=(2,2,2))
        # self.conv6_0 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = CovDropoutNormNonlin(240+120, 120, (3, 3, 3), (1, 1, 1))

        self.conv7_u = nn.ConvTranspose3d(120, 120, kernel_size=(2,2,2), stride=(2,2,2))
        # self.conv7_0 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv7 = CovDropoutNormNonlin(120+60, 60, (3, 3, 3), (1, 1, 1))

        self.conv8_u = nn.ConvTranspose3d(60, 60, kernel_size=(2,2,2), stride=(2,2,2))
        self.conv8 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))

        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        #decode
        # up5 = self.conv5_0(self.conv5_u(conv4))
        up5 = self.conv5_u(conv4_bottel)
        up5 = torch.cat([up5, conv4], dim=1)
        conv5 = self.conv5(up5)

        # up6 = self.conv6_0(self.conv6_u(conv5))
        up6 = self.conv6_u(conv5)
        up6 = torch.cat([up6, conv3], dim=1)
        conv6 = self.conv6(up6)

        # up7 = self.conv7_0(self.conv6_u(conv6))
        #TODO Big mistake!!!!
        # up7 = self.conv6_u(conv6)
        up7 = self.conv7_u(conv6)
        up7 = torch.cat([up7, conv2], dim=1)
        conv7 = self.conv7(up7)

        up8 = self.conv8_u(conv7)
        up8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(up8)
        # output layer
        conv9 = self.conv9(conv8)
        if self.softmax:
            output = F.softmax(conv9, dim=1)
        else:
            output = conv9
        return output, pool1


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class NoNewNetAsymm(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, softmax=False):
        super(NoNewNetAsymm, self).__init__()
        self.softmax = softmax
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        self.conv5_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv5_0 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))

        self.conv6_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv6_0 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = CovDropoutNormNonlin(240+120, 120, (3, 3, 3), (1, 1, 1))

        self.conv7_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv7_0 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv7 = CovDropoutNormNonlin(120+60, 60, (3, 3, 3), (1, 1, 1))

        # self.conv8_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv8 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))

        self.conv9 = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=(1, 1, 1))
        # Test multi-task learning
        self.conv9_1 = nn.Conv3d(in_channels=60, out_channels=1, kernel_size=(1, 1, 1))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        #decode
        # up5 = self.conv5_0(self.conv5_u(conv4))
        up5 = self.conv5_u(conv4_bottel)
        up5 = torch.cat([up5, conv4], dim=1)
        conv5 = self.conv5(up5)

        # up6 = self.conv6_0(self.conv6_u(conv5))
        up6 = self.conv6_u(conv5)
        up6 = torch.cat([up6, conv3], dim=1)
        conv6 = self.conv6(up6)

        # up7 = self.conv7_0(self.conv6_u(conv6))
        #TODO Big mistake!!!!
        # up7 = self.conv6_u(conv6)
        up7 = self.conv7_u(conv6)
        up7 = torch.cat([up7, conv2], dim=1)
        conv7 = self.conv7(up7)

        # up8 = self.conv8_u(conv7)
        # up8 = torch.cat([up8, conv1], dim=1)
        # conv8 = self.conv8(up8)
        # output layer
        conv9 = self.conv9(conv7)
        conv9_1 = self.conv9_1(conv7)
        if self.softmax:
            output = F.softmax(conv9, dim=1)
        else:
            output = conv9
        # return output, pool1, F.sigmoid(conv9_1)
        return output, conv1, conv7


class LDLSRNoNewNet(nn.Module):
    def __init__(self, input_channel, scale_factor, middle_channel=30, kernel_size=3, n_class=4, drop_rate=0):
        super(LDLSRNoNewNet, self).__init__()
        # self.ldl = Unet3DAsymm(input_dim=input_channel, output_dim=n_class, softmax=False).cuda()
        self.ldl = NoNewNetAsymm(input_dim=input_channel, output_dim=n_class,
                                      softmax=False)
        self.pbfr = PBFR(input_channel=middle_channel, kernel_size=kernel_size, out_channel=middle_channel,
                         scale_factor=scale_factor)
        self.espcn = ESPCN3D(scale_factor=scale_factor, input_channel=middle_channel,
                             kernel_size=kernel_size, n_class=n_class)

    def forward(self, input):
        # Notice: the input is the original size
        predict, subsampled_feature, _ = self.ldl(input)
        predict_dist = F.softmax(predict, dim=1)
        predict_dist_logsoftmax = F.log_softmax(predict, dim=1)
        predicted_entropy = entropy(predict_dist)
        prior_enhanced_feature = self.pbfr(predicted_entropy, predict_dist, subsampled_feature)
        sr_out = self.espcn(prior_enhanced_feature)

        return predict_dist_logsoftmax, predicted_entropy, sr_out


class LDLSRNoNewNetLast(nn.Module):
    '''
    use the feature of the last layer for "NoNewNetAsymm"
    '''
    def __init__(self, input_channel, scale_factor, middle_channel=60, kernel_size=3, n_class=4, drop_rate=0):
        super(LDLSRNoNewNetLast, self).__init__()
        # self.ldl = Unet3DAsymm(input_dim=input_channel, output_dim=n_class, softmax=False).cuda()
        self.ldl = NoNewNetAsymm(input_dim=input_channel, output_dim=n_class,
                                      softmax=False)
        self.pbfr = PBFR(input_channel=middle_channel, kernel_size=kernel_size, out_channel=middle_channel,
                         scale_factor=scale_factor)
        self.espcn = ESPCN3D(scale_factor=scale_factor, input_channel=middle_channel,
                             kernel_size=kernel_size, n_class=n_class)

    def forward(self, input):
        # Notice: the input is the original size
        predict, _,  subsampled_feature = self.ldl(input)
        predict_dist = F.softmax(predict, dim=1)
        predict_dist_logsoftmax = F.log_softmax(predict, dim=1)
        predicted_entropy = entropy(predict_dist)
        prior_enhanced_feature = self.pbfr(predicted_entropy, predict_dist, subsampled_feature)
        sr_out = self.espcn(prior_enhanced_feature)

        return predict_dist_logsoftmax, predicted_entropy, sr_out


class AttentionNoNewNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(AttentionNoNewNet, self).__init__()
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        self.conv5_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv5_0 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))

        self.conv6_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv6_0 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = CovDropoutNormNonlin(240+120, 120, (3, 3, 3), (1, 1, 1))

        self.conv7_u = Upsample(scale_factor=2, mode="trilinear")
        # self.conv7_0 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv7 = CovDropoutNormNonlin(120+60, 60, (3, 3, 3), (1, 1, 1))

        self.conv9 = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=(1, 1, 1))

        self.pbfr = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=middle_channel,
                         scale_factor=scale_factor)

        # self.pbfr = PBFRNew(input_channel=output_dim, kernel_size=kernel_size, out_channel=middle_channel,
        #                  scale_factor=scale_factor)

        self.conv10_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv10 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))
        self.conv11 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        #decode
        # up5 = self.conv5_0(self.conv5_u(conv4))
        up5 = self.conv5_u(conv4_bottel)
        up5 = torch.cat([up5, conv4], dim=1)
        conv5 = self.conv5(up5)

        # up6 = self.conv6_0(self.conv6_u(conv5))
        up6 = self.conv6_u(conv5)
        up6 = torch.cat([up6, conv3], dim=1)
        conv6 = self.conv6(up6)

        # up7 = self.conv7_0(self.conv6_u(conv6))
        #TODO Big mistake!!!!
        # up7 = self.conv6_u(conv6)
        up7 = self.conv7_u(conv6)
        up7 = torch.cat([up7, conv2], dim=1)
        conv7 = self.conv7(up7)

        # auxiliary branch
        conv9 = self.conv9(conv7)
        predict_dist = F.softmax(conv9, dim=1)
        # PAFR module
        predict_dist_logsoftmax = F.log_softmax(conv9, dim=1)
        predicted_entropy = entropy(predict_dist)
        prior_enhanced_feature = self.pbfr(predicted_entropy, predict_dist, conv7)

        # main branch
        up10 = self.conv10_u(prior_enhanced_feature)
        up10 = torch.cat([up10, conv1], dim=1)
        conv10 = self.conv10(up10)
        conv11 = self.conv11(conv10)

        return predict_dist_logsoftmax, predicted_entropy, conv11, predict_dist


class LSR(nn.Module):
    def __init__(self, input_dim=90, output_dim=None):
        super(LSR, self).__init__()
        self.conv10_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv10 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.conv11 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))
        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x, conv):
        up10 = self.conv10_u(x)
        up10 = torch.cat([up10, conv], dim=1)
        conv10 = self.conv10(up10)
        conv11 = self.conv11(conv10)
        return conv11


class P2ANet_old(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(P2ANet_old, self).__init__()
        self.ldl = NoNewNetAsymm(input_dim, output_dim, softmax=False)
        self.pbfr = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=middle_channel,
                         scale_factor=scale_factor)
        self.lsr = LSR(input_dim=90, output_dim=output_dim)

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        ldl_out, conv1, conv7 = self.ldl(x)
        predict_dist = F.softmax(ldl_out, dim=1)
        # PAFR module
        predict_dist_logsoftmax = F.log_softmax(ldl_out, dim=1)
        predicted_entropy = entropy(predict_dist)
        prior_enhanced_feature = self.pbfr(predicted_entropy, predict_dist, conv7)
        final_out = self.lsr(prior_enhanced_feature, conv1)

        return predict_dist_logsoftmax, predicted_entropy, final_out, predict_dist


class P2ANet(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(P2ANet, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part1
        self.conv5_u_1 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_1 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))
        self.aux_conv1 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)

        # encoder part2
        self.conv5_u_2 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv5_2 = CovDropoutNormNonlin(480 + 240+240, 320, (3, 3, 3), (1, 1, 1))
        self.conv6_u_2 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_2 = CovDropoutNormNonlin(320 + 120, 220, (3, 3, 3), (1, 1, 1))
        self.aux_conv2 = nn.Conv3d(in_channels=220, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=220,
                          scale_factor=scale_factor)
        # encoder part3
        self.conv5_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_3 = CovDropoutNormNonlin(480 + 240 , 240, (3, 3, 3), (1, 1, 1))
        self.conv6_u_3 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv6_3 = CovDropoutNormNonlin(240 + 120 + 220, 290, (3, 3, 3), (1, 1, 1))
        self.conv7_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_3 = CovDropoutNormNonlin(290 + 60, 180, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=180, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=180,
                          scale_factor=scale_factor)

        # label super-resolution
        self.conv8_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv8 = CovDropoutNormNonlin(180 + 30, 90, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=90, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part 1
        up5_1 = self.conv5_u_1(conv4_bottel)
        up5_1 = torch.cat([up5_1, conv4], dim=1)
        conv5_1 = self.conv5_1(up5_1)
        aux_conv1 = self.aux_conv1(conv5_1)

        predict_dist1 = F.softmax(aux_conv1, dim=1)
        predicted_entropy1 = entropy(predict_dist1)
        pbfr1 = self.pbfr1(predicted_entropy1, predict_dist1, conv5_1)
        predict_dist1_logsoftmax = F.log_softmax(aux_conv1, dim=1)

        # decode part 2
        up5_2 = self.conv5_u_2(conv4_bottel)
        up5_2 = torch.cat([up5_2, conv4, pbfr1], dim=1)
        conv5_2 = self.conv5_2(up5_2)
        up6_2 = self.conv6_u_2(conv5_2)
        up6_2 = torch.cat([up6_2, conv3], dim=1)
        conv6_2 = self.conv6_2(up6_2)
        aux_conv2 = self.aux_conv2(conv6_2)

        predict_dist2 = F.softmax(aux_conv2, dim=1)
        predicted_entropy2 = entropy(predict_dist2)
        pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv6_2)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part 3
        up5_3 = self.conv5_u_3(conv4_bottel)
        up5_3 = torch.cat([up5_3, conv4], dim=1)
        conv5_3 = self.conv5_3(up5_3)
        up6_3 = self.conv6_u_3(conv5_3)
        up6_3 = torch.cat([up6_3, conv3, pbfr2], dim=1)
        conv6_3 = self.conv6_3(up6_3)
        up7_3 = self.conv7_u_3(conv6_3)
        up7_3 = torch.cat([up7_3, conv2], dim=1)
        conv7_3 = self.conv7_3(up7_3)
        aux_conv3 = self.aux_conv3(conv7_3)

        predict_dist3 = F.softmax(aux_conv3, dim=1)
        predicted_entropy3 = entropy(predict_dist3)
        pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv7_3)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # label super-resolution
        up8 = self.conv8_u(pbfr3)
        up8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(up8)
        conv9 = self.conv9(conv8)

        aux_out = {'aux1': [predict_dist1_logsoftmax, predicted_entropy1],
                   'aux2': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux3': [predict_dist3_logsoftmax, predicted_entropy3]}

        return aux_out, conv9


# out of memory
class P2ANetNew(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(P2ANetNew, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # decoder part1
        self.conv5_u_1 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_1 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))
        self.aux_conv1 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)

        # decoder part2
        self.conv5_u_2 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv5_2 = CovDropoutNormNonlin(480 + 240+240, 320, (3, 3, 3), (1, 1, 1))
        self.conv6_u_2 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_2 = CovDropoutNormNonlin(320 + 120, 220, (3, 3, 3), (1, 1, 1))
        self.aux_conv2 = nn.Conv3d(in_channels=220, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=220,
                          scale_factor=scale_factor)
        # decoder part3
        self.conv5_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_3 = CovDropoutNormNonlin(480 + 240 , 240, (3, 3, 3), (1, 1, 1))
        self.conv6_u_3 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv6_3 = CovDropoutNormNonlin(240 + 120 + 220, 290, (3, 3, 3), (1, 1, 1))
        self.conv7_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_3 = CovDropoutNormNonlin(290 + 60, 180, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=180, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=180,
                          scale_factor=scale_factor)
        # decoder part4
        self.conv5_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_4 = CovDropoutNormNonlin(480+240, 240, (3,3,3), (1,1,1))
        self.conv6_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_4 = CovDropoutNormNonlin(240 + 120, 180, (3, 3, 3), (1, 1, 1))
        self.conv7_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_4 = CovDropoutNormNonlin(180 + 60 + 180, 210, (3, 3, 3), (1, 1, 1))
        self.conv8_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv8_4 = CovDropoutNormNonlin(210 + 30, 120, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=120, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part 1
        up5_1 = self.conv5_u_1(conv4_bottel)
        up5_1 = torch.cat([up5_1, conv4], dim=1)
        conv5_1 = self.conv5_1(up5_1)
        aux_conv1 = self.aux_conv1(conv5_1)

        predict_dist1 = F.softmax(aux_conv1, dim=1)
        predicted_entropy1 = entropy(predict_dist1)
        pbfr1 = self.pbfr1(predicted_entropy1, predict_dist1, conv5_1)
        predict_dist1_logsoftmax = F.log_softmax(aux_conv1, dim=1)

        # decode part 2
        up5_2 = self.conv5_u_2(conv4_bottel)
        up5_2 = torch.cat([up5_2, conv4, pbfr1], dim=1)
        conv5_2 = self.conv5_2(up5_2)
        up6_2 = self.conv6_u_2(conv5_2)
        up6_2 = torch.cat([up6_2, conv3], dim=1)
        conv6_2 = self.conv6_2(up6_2)
        aux_conv2 = self.aux_conv2(conv6_2)

        predict_dist2 = F.softmax(aux_conv2, dim=1)
        predicted_entropy2 = entropy(predict_dist2)
        pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv6_2)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part 3
        up5_3 = self.conv5_u_3(conv4_bottel)
        up5_3 = torch.cat([up5_3, conv4], dim=1)
        conv5_3 = self.conv5_3(up5_3)
        up6_3 = self.conv6_u_3(conv5_3)
        up6_3 = torch.cat([up6_3, conv3, pbfr2], dim=1)
        conv6_3 = self.conv6_3(up6_3)
        up7_3 = self.conv7_u_3(conv6_3)
        up7_3 = torch.cat([up7_3, conv2], dim=1)
        conv7_3 = self.conv7_3(up7_3)
        aux_conv3 = self.aux_conv3(conv7_3)

        predict_dist3 = F.softmax(aux_conv3, dim=1)
        predicted_entropy3 = entropy(predict_dist3)
        pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv7_3)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # decode part 4
        up5_4 = self.conv5_u_4(conv4_bottel)
        up5_4 = torch.cat([up5_4, conv4], dim=1)
        conv5_4 = self.conv5_4(up5_4)
        up6_4 = self.conv6_u_4(conv5_4)
        up6_4 = torch.cat([up6_4, conv3], dim=1)
        conv6_4 = self.conv6_4(up6_4)
        up7_4 = self.conv7_u_4(conv6_4)
        up7_4 = torch.cat([up7_4, conv2, pbfr3], dim=1)
        conv7_4 = self.conv7_4(up7_4)
        up8_4 = self.conv8_u_4(conv7_4)
        up8_4 = torch.cat([up8_4, conv1], dim=1)
        conv8_4 = self.conv8_4(up8_4)
        conv9 = self.conv9(conv8_4)

        # # label super-resolution
        # up8 = self.conv8_u(pbfr3)
        # up8 = torch.cat([up8, conv1], dim=1)
        # conv8 = self.conv8(up8)
        # conv9 = self.conv9(conv8)

        aux_out = {'aux1': [predict_dist1_logsoftmax, predicted_entropy1],
                   'aux2': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux3': [predict_dist3_logsoftmax, predicted_entropy3]}

        return aux_out, conv9


class PACNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(PACNet, self).__init__()
        self.scale = scale_factor
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part1
        self.conv5_u_1 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_1 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))
        self.aux_conv1 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)

        # encoder part2
        self.conv5_u_2 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv5_2 = CovDropoutNormNonlin(480 + 240+240, 320, (3, 3, 3), (1, 1, 1))
        self.conv6_u_2 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_2 = CovDropoutNormNonlin(320 + 120, 220, (3, 3, 3), (1, 1, 1))
        self.aux_conv2 = nn.Conv3d(in_channels=220, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=220,
                          scale_factor=scale_factor)
        # encoder part3
        self.conv5_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_3 = CovDropoutNormNonlin(480 + 240 , 240, (3, 3, 3), (1, 1, 1))
        self.conv6_u_3 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv6_3 = CovDropoutNormNonlin(240 + 120 + 220, 290, (3, 3, 3), (1, 1, 1))
        self.conv7_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_3 = CovDropoutNormNonlin(290 + 60, 180, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=180, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=180,
                          scale_factor=scale_factor)

        # label distribution pyramid fusion
        self.conv8_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv8 = CovDropoutNormNonlin(180 + 30, 90, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=90, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part 1
        up5_1 = self.conv5_u_1(conv4_bottel)
        up5_1 = torch.cat([up5_1, conv4], dim=1)
        conv5_1 = self.conv5_1(up5_1)
        aux_conv1 = self.aux_conv1(conv5_1)

        predict_dist1 = F.softmax(aux_conv1, dim=1)
        predicted_entropy1 = entropy(predict_dist1)
        pbfr1 = self.pbfr1(predicted_entropy1, predict_dist1, conv5_1)
        predict_dist1_logsoftmax = F.log_softmax(aux_conv1, dim=1)

        # decode part 2
        up5_2 = self.conv5_u_2(conv4_bottel)
        up5_2 = torch.cat([up5_2, conv4, pbfr1], dim=1)
        conv5_2 = self.conv5_2(up5_2)
        up6_2 = self.conv6_u_2(conv5_2)
        up6_2 = torch.cat([up6_2, conv3], dim=1)
        conv6_2 = self.conv6_2(up6_2)
        aux_conv2 = self.aux_conv2(conv6_2)

        predict_dist2 = F.softmax(aux_conv2, dim=1)
        predicted_entropy2 = entropy(predict_dist2)
        pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv6_2)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # label calibration between adjacent layers of the pyramid
        kernel = torch.ones(1, 1, self.scale, self.scale, self.scale).cuda()
        aux2_stage2_1 = F.conv3d(aux_conv2.permute(1, 0, 2, 3, 4), kernel, stride=self.scale, padding=0)
        aux2_stage2_1 = aux2_stage2_1.permute(1, 0, 2,3,4)
        # decode part 3
        up5_3 = self.conv5_u_3(conv4_bottel)
        up5_3 = torch.cat([up5_3, conv4], dim=1)
        conv5_3 = self.conv5_3(up5_3)
        up6_3 = self.conv6_u_3(conv5_3)
        up6_3 = torch.cat([up6_3, conv3, pbfr2], dim=1)
        conv6_3 = self.conv6_3(up6_3)
        up7_3 = self.conv7_u_3(conv6_3)
        up7_3 = torch.cat([up7_3, conv2], dim=1)
        conv7_3 = self.conv7_3(up7_3)
        aux_conv3 = self.aux_conv3(conv7_3)

        predict_dist3 = F.softmax(aux_conv3, dim=1)
        predicted_entropy3 = entropy(predict_dist3)
        pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv7_3)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # kernel2 = torch.ones(1, 1, self.scale*2, self.scale*2, self.scale*2).cuda()
        aux3_stage3_2 = F.conv3d(aux_conv3.permute(1, 0, 2, 3, 4), kernel, stride=self.scale, padding=0)
        # aux3_stage3_1 = F.conv3d(aux_conv3.permute(1, 0, 2, 3, 4), kernel2, stride=self.scale*2, padding=0)
        aux3_stage3_2 = aux3_stage3_2.permute(1, 0, 2,3,4)
        # aux3_stage3_1 = aux3_stage3_1.permute(1,0, 2,3,4)
        # label super-resolution
        up8 = self.conv8_u(pbfr3)
        up8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(up8)
        conv9 = self.conv9(conv8)

        out4_3 = F.conv3d(conv9.permute(1, 0, 2, 3, 4), kernel, stride=self.scale, padding=0)
        out4_3 = out4_3.permute(1, 0, 2,3,4)

        aux_out = {'aux1': [predict_dist1_logsoftmax, predicted_entropy1],
                   'aux2': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux3': [predict_dist3_logsoftmax, predicted_entropy3]}

        aux_out_trans = {'aux2_1': F.softmax(aux2_stage2_1, dim=1),
                         'aux3_2': F.softmax(aux3_stage3_2, dim=1),
                         'aux4_3': F.softmax(out4_3, dim=1)}

        return aux_out, conv9, aux_out_trans


class UNetLDL(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(UNetLDL, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part
        # self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=480,
        #                   scale_factor=scale_factor)
        self.conv5_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_4 = CovDropoutNormNonlin(480+240, 240, (3,3,3), (1,1,1))
        self.aux_conv2 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)
        self.conv6_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_4 = CovDropoutNormNonlin(240 + 120, 120, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=120, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=120,
                          scale_factor=scale_factor)
        self.conv7_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_4 = CovDropoutNormNonlin(120 + 60 , 60, (3, 3, 3), (1, 1, 1))
        self.aux_conv4 = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr4 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=60,
                          scale_factor=scale_factor)
        self.conv8_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv8_4 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part stage1
        up5_4 = self.conv5_u_4(conv4_bottel)
        up5_4 = torch.cat([up5_4, conv4], dim=1)
        conv5_4 = self.conv5_4(up5_4)
        aux_conv2 = self.aux_conv2(conv5_4)

        predict_dist2 = F.softmax(aux_conv2, dim=1)
        predicted_entropy2 = entropy(predict_dist2)
        pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv5_4)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part stage2
        up6_4 = self.conv6_u_4(pbfr2)
        up6_4 = torch.cat([up6_4, conv3], dim=1)
        conv6_4 = self.conv6_4(up6_4)
        aux_conv3 = self.aux_conv3(conv6_4)

        predict_dist3 = F.softmax(aux_conv3, dim=1)
        predicted_entropy3 = entropy(predict_dist3)
        pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv6_4)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # decode part stage3
        up7_4 = self.conv7_u_4(pbfr3)
        up7_4 = torch.cat([up7_4, conv2], dim=1)
        conv7_4 = self.conv7_4(up7_4)
        aux_conv4 = self.aux_conv4(conv7_4)

        predict_dist4 = F.softmax(aux_conv4, dim=1)
        predicted_entropy4 = entropy(predict_dist4)
        pbfr4 = self.pbfr4(predicted_entropy4, predict_dist4, conv7_4)
        predict_dist4_logsoftmax = F.log_softmax(aux_conv4, dim=1)

        up8_4 = self.conv8_u_4(pbfr4)
        up8_4 = torch.cat([up8_4, conv1], dim=1)
        conv8_4 = self.conv8_4(up8_4)
        conv9 = self.conv9(conv8_4)

        aux_out = {'aux1': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux2': [predict_dist3_logsoftmax, predicted_entropy3],
                   'aux3': [predict_dist4_logsoftmax, predicted_entropy4]}

        return aux_out, conv9


class P2ANetWithoutPBFR(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(P2ANetWithoutPBFR, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part1
        self.conv5_u_1 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_1 = CovDropoutNormNonlin(480+240, 240, (3, 3, 3), (1, 1, 1))
        self.aux_conv1 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        # self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
        #                   scale_factor=scale_factor)

        # encoder part2
        self.conv5_u_2 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv5_2 = CovDropoutNormNonlin(480 + 240, 320, (3, 3, 3), (1, 1, 1))
        self.conv6_u_2 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_2 = CovDropoutNormNonlin(320 + 120, 220, (3, 3, 3), (1, 1, 1))
        self.aux_conv2 = nn.Conv3d(in_channels=220, out_channels=output_dim, kernel_size=(1, 1, 1))
        # self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=220,
        #                   scale_factor=scale_factor)
        # encoder part3
        self.conv5_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_3 = CovDropoutNormNonlin(480 + 240 , 240, (3, 3, 3), (1, 1, 1))
        self.conv6_u_3 = Upsample(scale_factor=2, mode="trilinear")
        # Three parts concatenate: early features+ conv5_u_2_features + pbfr1_features
        self.conv6_3 = CovDropoutNormNonlin(240 + 120, 290, (3, 3, 3), (1, 1, 1))
        self.conv7_u_3 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_3 = CovDropoutNormNonlin(290 + 60, 180, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=180, out_channels=output_dim, kernel_size=(1, 1, 1))
        # self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=180,
        #                   scale_factor=scale_factor)

        # label super-resolution
        self.conv8_u = Upsample(scale_factor=2, mode="trilinear")
        self.conv8 = CovDropoutNormNonlin(180 + 30, 90, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=90, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part 1
        up5_1 = self.conv5_u_1(conv4_bottel)
        up5_1 = torch.cat([up5_1, conv4], dim=1)
        conv5_1 = self.conv5_1(up5_1)
        aux_conv1 = self.aux_conv1(conv5_1)

        # predict_dist1 = F.softmax(aux_conv1, dim=1)
        # predicted_entropy1 = entropy(predict_dist1)
        # pbfr1 = self.pbfr1(predicted_entropy1, predict_dist1, conv5_1)
        predict_dist1_logsoftmax = F.log_softmax(aux_conv1, dim=1)

        # decode part 2
        up5_2 = self.conv5_u_2(conv4_bottel)
        up5_2 = torch.cat([up5_2, conv4], dim=1)
        conv5_2 = self.conv5_2(up5_2)
        up6_2 = self.conv6_u_2(conv5_2)
        up6_2 = torch.cat([up6_2, conv3], dim=1)
        conv6_2 = self.conv6_2(up6_2)
        aux_conv2 = self.aux_conv2(conv6_2)

        # predict_dist2 = F.softmax(aux_conv2, dim=1)
        # predicted_entropy2 = entropy(predict_dist2)
        # pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv6_2)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part 3
        up5_3 = self.conv5_u_3(conv4_bottel)
        up5_3 = torch.cat([up5_3, conv4], dim=1)
        conv5_3 = self.conv5_3(up5_3)
        up6_3 = self.conv6_u_3(conv5_3)
        up6_3 = torch.cat([up6_3, conv3], dim=1)
        conv6_3 = self.conv6_3(up6_3)
        up7_3 = self.conv7_u_3(conv6_3)
        up7_3 = torch.cat([up7_3, conv2], dim=1)
        conv7_3 = self.conv7_3(up7_3)
        aux_conv3 = self.aux_conv3(conv7_3)

        # predict_dist3 = F.softmax(aux_conv3, dim=1)
        # predicted_entropy3 = entropy(predict_dist3)
        # pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv7_3)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # label super-resolution
        up8 = self.conv8_u(conv7_3)
        up8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(up8)
        conv9 = self.conv9(conv8)

        predicted_entropy1=0
        predicted_entropy2=0
        predicted_entropy3=0

        aux_out = {'aux1': [predict_dist1_logsoftmax, predicted_entropy1],
                   'aux2': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux3': [predict_dist3_logsoftmax, predicted_entropy3]}

        return aux_out, conv9


class EncoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, drop_rate=0):
        super(EncoderModule, self).__init__()
        self.drop_rate = drop_rate
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.dropout0 = nn.Dropout3d(drop_rate, inplace=True)
        self.batchnorm0 = nn.BatchNorm3d(out_channels, affine=True)
        self.relu0 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        if self.drop_rate>0:
            x = self.dropout0(x)
        x = self.batchnorm0(x)
        x = self.relu0(x)

        return x


class DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm0 = nn.BatchNorm3d(out_channels, affine=True)
        self.relu0 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.relu0(x)

        return x


class UNet3D(nn.Module):
    def __init__(self, input_dim=None, output_dim=None):
        super(UNet3D, self).__init__()
        self.conv1 = EncoderModule(input_dim, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = EncoderModule(64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3_1 = EncoderModule(128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = EncoderModule(256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4_1 = EncoderModule(256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = EncoderModule(512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5_1 = EncoderModule(512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = EncoderModule(512, out_channels=512, kernel_size=3, padding=1)

        self.deconv1_1 = DecoderModule(512, out_channels=512, kernel_size=2, stride=2)
        self.deconv1_2 = EncoderModule(512+512, out_channels=256, kernel_size=3, padding=1)

        self.deconv2_1 = DecoderModule(256, out_channels=256, kernel_size=2, stride=2)
        self.deconv2_2 = EncoderModule(256 + 256, out_channels=128, kernel_size=3, padding=1)

        self.deconv3_1 = DecoderModule(128, out_channels=128, kernel_size=2, stride=2)
        self.deconv3_2 = EncoderModule(128 + 128, out_channels=64, kernel_size=3, padding=1)

        self.deconv4_1 = DecoderModule(64, out_channels=64, kernel_size=2, stride=2)
        self.deconv4_2 = EncoderModule(64 + 64, out_channels=32, kernel_size=3, padding=1)

        self.predicted_prob = nn.Conv3d(in_channels=32, out_channels=output_dim, kernel_size=1, stride=1)
        # auxiliary prediction 0
        self.aux0_conv = nn.Conv3d(in_channels=256, out_channels=output_dim, kernel_size=1, stride=1)
        self.aux0_deconv1 = nn.ConvTranspose3d(in_channels=output_dim, out_channels=output_dim,
                                        kernel_size=2, stride=2)
        self.aux0_deconv2 = nn.ConvTranspose3d(in_channels=output_dim, out_channels=output_dim,
                                               kernel_size=2, stride=2)
        self.aux0_prob = nn.ConvTranspose3d(in_channels=output_dim, out_channels=output_dim,
                                               kernel_size=2, stride=2)
        # auxiliary prediction 1
        self.aux1_conv = nn.Conv3d(in_channels=128, out_channels=output_dim, kernel_size=1, stride=1)
        self.aux1_deconv1 = nn.ConvTranspose3d(in_channels=output_dim, out_channels=output_dim,
                                               kernel_size=2, stride=2)
        self.aux1_prob = nn.ConvTranspose3d(in_channels=output_dim, out_channels=output_dim,
                                               kernel_size=2, stride=2)
        # auxiliary prediction 2
        self.aux2_conv = nn.Conv3d(in_channels=64, out_channels=output_dim, kernel_size=1, stride=1)
        self.aux2_prob = nn.ConvTranspose3d(in_channels=output_dim, out_channels=output_dim,
                                               kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        deconv1_1 = self.deconv1_1(conv5_2)
        cat1 = torch.cat([deconv1_1, conv4_2], dim=1)
        deconv1_2 = self.deconv1_2(cat1)

        deconv2_1 = self.deconv2_1(deconv1_2)
        cat2 = torch.cat([deconv2_1, conv3_2], dim=1)
        deconv2_2 = self.deconv2_2(cat2)

        deconv3_1 = self.deconv3_1(deconv2_2)
        cat3 = torch.cat([deconv3_1, conv2], dim=1)
        deconv3_2  = self.deconv3_2(cat3)

        deconv4_1 = self.deconv4_1(deconv3_2)
        cat4 = torch.cat([deconv4_1, conv1], dim=1)
        deconv4_2 = self.deconv4_2(cat4)

        predicted_prob = self.predicted_prob(deconv4_2)

        aux0_conv = self.aux0_conv(deconv1_2)
        aux0_deconv1 = self.aux0_deconv1(aux0_conv)
        aux0_deconv2 = self.aux0_deconv2(aux0_deconv1)
        aux0_prob = self.aux0_prob(aux0_deconv2)

        aux1_conv = self.aux1_conv(deconv2_2)
        aux1_deconv1 = self.aux1_deconv1(aux1_conv)
        aux1_prob = self.aux1_prob(aux1_deconv1)

        aux2_conv = self.aux2_conv(deconv3_2)
        aux2_prob = self.aux2_prob(aux2_conv)

        return predicted_prob, aux0_prob, aux1_prob, aux2_prob


class UNetLDLWithoutPPA(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(UNetLDLWithoutPPA, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part
        # self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=480,
        #                   scale_factor=scale_factor)
        self.conv5_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_4 = CovDropoutNormNonlin(480+240, 240, (3,3,3), (1,1,1))
        self.aux_conv2 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)
        self.conv6_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_4 = CovDropoutNormNonlin(240 + 120, 120, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=120, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=120,
                          scale_factor=scale_factor)
        self.conv7_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_4 = CovDropoutNormNonlin(120 + 60 , 60, (3, 3, 3), (1, 1, 1))
        self.aux_conv4 = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr4 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=60,
                          scale_factor=scale_factor)
        self.conv8_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv8_4 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part stage1
        up5_4 = self.conv5_u_4(conv4_bottel)
        up5_4 = torch.cat([up5_4, conv4], dim=1)
        conv5_4 = self.conv5_4(up5_4)
        aux_conv2 = self.aux_conv2(conv5_4)

        # predict_dist2 = F.softmax(aux_conv2, dim=1)
        # predicted_entropy2 = entropy(predict_dist2)
        # pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv5_4)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part stage2
        up6_4 = self.conv6_u_4(conv5_4)
        up6_4 = torch.cat([up6_4, conv3], dim=1)
        conv6_4 = self.conv6_4(up6_4)
        aux_conv3 = self.aux_conv3(conv6_4)

        # predict_dist3 = F.softmax(aux_conv3, dim=1)
        # predicted_entropy3 = entropy(predict_dist3)
        # pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv6_4)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # decode part stage3
        up7_4 = self.conv7_u_4(conv6_4)
        up7_4 = torch.cat([up7_4, conv2], dim=1)
        conv7_4 = self.conv7_4(up7_4)
        aux_conv4 = self.aux_conv4(conv7_4)

        # predict_dist4 = F.softmax(aux_conv4, dim=1)
        # predicted_entropy4 = entropy(predict_dist4)
        # pbfr4 = self.pbfr4(predicted_entropy4, predict_dist4, conv7_4)
        predict_dist4_logsoftmax = F.log_softmax(aux_conv4, dim=1)

        up8_4 = self.conv8_u_4(conv7_4)
        up8_4 = torch.cat([up8_4, conv1], dim=1)
        conv8_4 = self.conv8_4(up8_4)
        conv9 = self.conv9(conv8_4)

        aux_out = {'aux1': [predict_dist2_logsoftmax, predict_dist2_logsoftmax],
                   'aux2': [predict_dist3_logsoftmax, predict_dist2_logsoftmax],
                   'aux3': [predict_dist4_logsoftmax, predict_dist2_logsoftmax]}

        return aux_out, conv9


# ablation model
class UNetLDLN1(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(UNetLDLN1, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part
        # self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=480,
        #                   scale_factor=scale_factor)
        self.conv5_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_4 = CovDropoutNormNonlin(480+240, 240, (3,3,3), (1,1,1))
        self.aux_conv2 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)
        self.conv6_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_4 = CovDropoutNormNonlin(240 + 120, 120, (3, 3, 3), (1, 1, 1))
        # self.aux_conv3 = nn.Conv3d(in_channels=120, out_channels=output_dim, kernel_size=(1, 1, 1))
        # self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=120,
        #                   scale_factor=scale_factor)
        self.conv7_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_4 = CovDropoutNormNonlin(120 + 60 , 60, (3, 3, 3), (1, 1, 1))
        # self.aux_conv4 = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=(1, 1, 1))
        # self.pbfr4 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=60,
        #                   scale_factor=scale_factor)
        self.conv8_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv8_4 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part stage1
        up5_4 = self.conv5_u_4(conv4_bottel)
        up5_4 = torch.cat([up5_4, conv4], dim=1)
        conv5_4 = self.conv5_4(up5_4)
        aux_conv2 = self.aux_conv2(conv5_4)

        predict_dist2 = F.softmax(aux_conv2, dim=1)
        predicted_entropy2 = entropy(predict_dist2)
        pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv5_4)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part stage2
        up6_4 = self.conv6_u_4(pbfr2)
        up6_4 = torch.cat([up6_4, conv3], dim=1)
        conv6_4 = self.conv6_4(up6_4)
        # aux_conv3 = self.aux_conv3(conv6_4)

        # predict_dist3 = F.softmax(aux_conv3, dim=1)
        # predicted_entropy3 = entropy(predict_dist3)
        # pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv6_4)
        # predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # decode part stage3
        up7_4 = self.conv7_u_4(conv6_4)
        up7_4 = torch.cat([up7_4, conv2], dim=1)
        conv7_4 = self.conv7_4(up7_4)
        # aux_conv4 = self.aux_conv4(conv7_4)

        # predict_dist4 = F.softmax(aux_conv4, dim=1)
        # predicted_entropy4 = entropy(predict_dist4)
        # pbfr4 = self.pbfr4(predicted_entropy4, predict_dist4, conv7_4)
        # predict_dist4_logsoftmax = F.log_softmax(aux_conv4, dim=1)

        up8_4 = self.conv8_u_4(conv7_4)
        up8_4 = torch.cat([up8_4, conv1], dim=1)
        conv8_4 = self.conv8_4(up8_4)
        conv9 = self.conv9(conv8_4)

        predict_dist3_logsoftmax = 0
        predict_dist4_logsoftmax = 0
        predicted_entropy3 = 0
        predicted_entropy4 = 0

        aux_out = {'aux1': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux2': [predict_dist3_logsoftmax, predicted_entropy3],
                   'aux3': [predict_dist4_logsoftmax, predicted_entropy4]}

        return aux_out, conv9


class UNetLDLN2(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, scale_factor=2, middle_channel=60, kernel_size=3):
        super(UNetLDLN2, self).__init__()
        # encoder part
        self.conv1 = CovDropoutNormNonlin(input_dim, 30, (3, 3, 3), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = CovDropoutNormNonlin(30, 60, (3, 3, 3), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = CovDropoutNormNonlin(60, 120, (3, 3, 3), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        self.conv4 = CovDropoutNormNonlin(120, 240, (3, 3, 3), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2,2,2))

        self.conv4_bottel = CovDropoutNormNonlin(240, 480, (3, 3, 3), (1, 1, 1))

        # encoder part
        # self.pbfr1 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=480,
        #                   scale_factor=scale_factor)
        self.conv5_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv5_4 = CovDropoutNormNonlin(480+240, 240, (3,3,3), (1,1,1))
        self.aux_conv2 = nn.Conv3d(in_channels=240, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr2 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=240,
                          scale_factor=scale_factor)
        self.conv6_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv6_4 = CovDropoutNormNonlin(240 + 120, 120, (3, 3, 3), (1, 1, 1))
        self.aux_conv3 = nn.Conv3d(in_channels=120, out_channels=output_dim, kernel_size=(1, 1, 1))
        self.pbfr3 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=120,
                          scale_factor=scale_factor)
        self.conv7_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv7_4 = CovDropoutNormNonlin(120 + 60 , 60, (3, 3, 3), (1, 1, 1))
        # self.aux_conv4 = nn.Conv3d(in_channels=60, out_channels=output_dim, kernel_size=(1, 1, 1))
        # self.pbfr4 = PBFR(input_channel=output_dim, kernel_size=kernel_size, out_channel=60,
        #                   scale_factor=scale_factor)
        self.conv8_u_4 = Upsample(scale_factor=2, mode="trilinear")
        self.conv8_4 = CovDropoutNormNonlin(60 + 30, 30, (3, 3, 3), (1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=30, out_channels=output_dim, kernel_size=(1, 1, 1))

        init = InitWeights_He(1e-2)
        self.apply(init)

    def forward(self, x):
        # Encode part
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        # bottle neck layer
        conv4_bottel = self.conv4_bottel(pool4)

        # decode part stage1
        up5_4 = self.conv5_u_4(conv4_bottel)
        up5_4 = torch.cat([up5_4, conv4], dim=1)
        conv5_4 = self.conv5_4(up5_4)
        aux_conv2 = self.aux_conv2(conv5_4)

        predict_dist2 = F.softmax(aux_conv2, dim=1)
        predicted_entropy2 = entropy(predict_dist2)
        pbfr2 = self.pbfr2(predicted_entropy2, predict_dist2, conv5_4)
        predict_dist2_logsoftmax = F.log_softmax(aux_conv2, dim=1)

        # decode part stage2
        up6_4 = self.conv6_u_4(pbfr2)
        up6_4 = torch.cat([up6_4, conv3], dim=1)
        conv6_4 = self.conv6_4(up6_4)
        aux_conv3 = self.aux_conv3(conv6_4)

        predict_dist3 = F.softmax(aux_conv3, dim=1)
        predicted_entropy3 = entropy(predict_dist3)
        pbfr3 = self.pbfr3(predicted_entropy3, predict_dist3, conv6_4)
        predict_dist3_logsoftmax = F.log_softmax(aux_conv3, dim=1)

        # decode part stage3
        up7_4 = self.conv7_u_4(pbfr3)
        up7_4 = torch.cat([up7_4, conv2], dim=1)
        conv7_4 = self.conv7_4(up7_4)
        # aux_conv4 = self.aux_conv4(conv7_4)

        # predict_dist4 = F.softmax(aux_conv4, dim=1)
        # predicted_entropy4 = entropy(predict_dist4)
        # pbfr4 = self.pbfr4(predicted_entropy4, predict_dist4, conv7_4)
        # predict_dist4_logsoftmax = F.log_softmax(aux_conv4, dim=1)

        up8_4 = self.conv8_u_4(conv7_4)
        up8_4 = torch.cat([up8_4, conv1], dim=1)
        conv8_4 = self.conv8_4(up8_4)
        conv9 = self.conv9(conv8_4)

        predict_dist4_logsoftmax = 0
        predicted_entropy4 = 0

        aux_out = {'aux1': [predict_dist2_logsoftmax, predicted_entropy2],
                   'aux2': [predict_dist3_logsoftmax, predicted_entropy3],
                   'aux3': [predict_dist4_logsoftmax, predicted_entropy4]}

        return aux_out, conv9


# Test Pre-trained models(ResNet Seg)
def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
                input_dim=opt.input_dim)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)

    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    # exclude the first layer
    first_layer = ["module.conv1.weight", "module.conv1.bias"]
    if opt.phase != 'test' and opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        pretrain_dict = dict()
        for k, v in pretrain['state_dict'].items():
            if k in net_dict.keys() and k not in first_layer:
                pretrain_dict[k] = v
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        print("Successfully Fine-tuned with DenseNet head!")
        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0 or pname=="module.conv1.weight":
                    new_parameters.append(p)
                    # break
        # for pname, p in model.named_parameters():
        #     for layer_name in opt.new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break
        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()


# calculate the Parameters and FLOPs of the model
def calculate_params_flops(net, input_shape):
    '''
    The function calculate the Parameters and Flops of the model
    net: the target model
    input_shape: input shape of the model: e.g. (1,4,96,96,96), batch*C*W*H*D
    '''
    from ptflops import get_model_complexity_info
    net.cuda()
    macs, params = get_model_complexity_info(net, input_shape, as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class LDLESPCN(nn.Module):
    '''
    Combined two stages of processing.
    '''
    def __init__(self, input_channel, scale_factor, kernel_size=3, n_class=4):
        super(LDLESPCN, self).__init__()
        self.ldl = Unet3D(input_dim=input_channel, output_dim=n_class, softmax=False)
        self.espcn = ESPCN3D(scale_factor=scale_factor, input_channel=input_channel+n_class,
                             kernel_size=kernel_size, n_class=n_class)

    def forward(self, *input):
        predict, _ = self.ldl(input[0])
        predict_dist_logsoftmax = F.log_softmax(predict, dim=1)
        # keep the data coinsistency with the stage2
        predict_dist = F.softmax(predict, dim=1)
        stage2_input = torch.cat((input[1], predict_dist), dim=1)
        sr_out = self.espcn(stage2_input)
        return predict_dist_logsoftmax, sr_out



if __name__ == "__main__":
    
    net_ldl = UNetLDL()# 536.89 GMac  21.74 M 
    nonewnet = NoNewNet(input_dim=4, output_dim=4) # 513.31 GMac 20.69 M 
    from Vnet import VNet
    from ESPNet import ESPNet
    espnet = ESPNet(classes=4, channels=4)
    vnet = VNet(in_channels=4) # 789.09 GMac 45.61 M
    models = [net_ldl, nonewnet, vnet, espnet]
    # input_dim = (4, 128, 128, 128)
    input_dim = (4, 96, 96, 96)
    for i in range(len(models)):
        model = models[i]
        calculate_params_flops(model, input_dim)

    # with torch.cuda.device(0):
    #     net = Unet3D(input_dim=3, output_dim=4).cuda()
    #     macs, params = get_model_complexity_info(net, (3, 96, 96, 96), as_strings=True, print_per_layer_stat=False)
    #     macs1, params1 = get_model_complexity_info(net, (3, 48, 48, 48), as_strings=True, print_per_layer_stat=False)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs1))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params1))
    # a = 1
