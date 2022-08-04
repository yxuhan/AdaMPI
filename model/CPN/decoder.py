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
import torch.nn.functional as F


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedConv, self).__init__()
        self.pad = nn.ReflectionPad2d(1)

        self.conv2d = nn.Conv2d(in_channels, out_channels, 3)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, 3)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feat):
        feat = self.pad(feat)
        x = self.conv2d(feat)
        mask = self.mask_conv2d(feat)
        return x * self.sigmoid(mask)


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedConvBlock, self).__init__()
        self.gated_conv = GatedConv(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, feat):
        x = self.gated_conv(feat)
        x = self.bn(x)
        x = self.nonlin(x)
        return x


def conv(in_planes, out_planes, kernel_size, instancenorm=False):
    if instancenorm:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    return m


class DepthDecoder(nn.Module):
    def tuple_to_str(self, key_tuple):
        key_str = '-'.join(str(key_tuple))
        return key_str

    def __init__(self, num_ch_enc,
                 use_alpha=False, scales=range(4), num_output_channels=4,
                 use_skips=True, **kwargs):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.use_alpha = use_alpha

        final_enc_out_channels = num_ch_enc[-1]
        self.downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_down1 = conv(final_enc_out_channels, 512, 1, False)
        self.conv_down2 = conv(512, 256, 3, False)
        self.conv_up1 = conv(256, 256, 3, False)
        self.conv_up2 = conv(256, final_enc_out_channels, 1, False)

        self.num_ch_enc = num_ch_enc
        print("num_ch_enc=", num_ch_enc)
        self.num_ch_enc = [x + 2 for x in self.num_ch_enc]
        self.num_ch_dec = np.array([12, 24, 48, 96, 192])
        # self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[self.tuple_to_str(("upconv", i, 0))] = GatedConvBlock(num_ch_in, num_ch_out)
            print("upconv_{}_{}".format(i, 0), num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[self.tuple_to_str(("upconv", i, 1))] = GatedConvBlock(num_ch_in, num_ch_out)
            print("upconv_{}_{}".format(i, 1), num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[self.tuple_to_str(("dispconv", s))] = GatedConv(self.num_ch_dec[s], self.num_output_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, feature_mask):
        B, S, _, _ = feature_mask.size()
        # extension of encoder to increase receptive field
        encoder_out = input_features[-1]
        conv_down1 = self.conv_down1(self.downsample(encoder_out))
        conv_down2 = self.conv_down2(self.downsample(conv_down1))
        conv_up1 = self.conv_up1(self.upsample(conv_down2))
        conv_up2 = self.conv_up2(self.upsample(conv_up1))

        # repeat / reshape features
        _, C_feat, H_feat, W_feat = conv_up2.size()
        cum_mask = torch.cumsum(feature_mask, dim=1)  # [B,S,H,W]
        inpaint_mask = torch.cat([torch.zeros_like(cum_mask[:, -1:, :, :]), cum_mask[:, :-1, :, :]], dim=1)  # [B,S,H,W]
        context_mask = 1 - inpaint_mask  # [B,S,H,W]

        cur_context_mask = F.adaptive_avg_pool2d(context_mask, (H_feat, W_feat)).unsqueeze(2)
        cur_feature_mask = F.adaptive_avg_pool2d(feature_mask, (H_feat, W_feat)).unsqueeze(2)
        conv_up2 = conv_up2.unsqueeze(1).repeat(1, S, 1, 1, 1)
        conv_up2 = torch.cat([conv_up2 * cur_context_mask, cur_context_mask, cur_feature_mask], dim=2)  # [B,S,C+2,H,W]
        conv_up2 = conv_up2.reshape(-1, C_feat + 2, H_feat, W_feat)  # [BxS,C+2,H,W]

        # repeat / reshape features
        for i, feat in enumerate(input_features):
            _, C_feat, H_feat, W_feat = feat.size()
            cur_context_mask = F.adaptive_avg_pool2d(context_mask, (H_feat, W_feat)).unsqueeze(2)
            cur_feature_mask = F.adaptive_avg_pool2d(feature_mask, (H_feat, W_feat)).unsqueeze(2)
            feat = feat.unsqueeze(1).repeat(1, S, 1, 1, 1)
            feat = torch.cat([feat * cur_context_mask, cur_context_mask, cur_feature_mask], dim=2)  # [B,S,C+2,H,W]
            input_features[i] = feat.reshape(-1, C_feat + 2, H_feat, W_feat)  # [BxS,C+2,H,W]

        outputs = []
        x = conv_up2
        for i in range(4, -1, -1):
            x = self.convs[self.tuple_to_str(("upconv", i, 0))](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[self.tuple_to_str(("upconv", i, 1))](x)
            if i in self.scales:
                output = self.convs[self.tuple_to_str(("dispconv", i))](x)
                H_mpi, W_mpi = output.size(2), output.size(3)
                cur_mask = F.adaptive_avg_pool2d(cum_mask, (H_mpi, W_mpi)).unsqueeze(2)
                mpi = output.view(B, S, 4, H_mpi, W_mpi)
                mpi_rgb = self.sigmoid(mpi[:, :, 0:3, :, :])
                if self.use_alpha:
                    mpi_sigma = self.sigmoid(mpi[:, :, 3:, :, :]) * cur_mask
                else:
                    mpi_sigma = torch.relu(mpi[:, :, 3:, :, :] * cur_mask) + 1e-4
                outputs.append(torch.cat((mpi_rgb, mpi_sigma), dim=2))
        return outputs[::-1]
        