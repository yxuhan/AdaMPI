import torch
import torch.nn as nn
import torch.nn.functional as F


class MPIPredictor(nn.Module):
    def __init__(
        self,
        width=384,
        height=256,
        num_planes=64,
    ):
        super(MPIPredictor, self).__init__()
        self.num_planes = num_planes
        disp_range = [0.001, 1]
        self.far, self.near = disp_range

        H_tgt, W_tgt = height, width
        ctx_spatial_scale = 4
        self.low_res_size = (int(H_tgt / ctx_spatial_scale), int(W_tgt / ctx_spatial_scale))

        # -----------------------
        # CPN Encoder
        # -----------------------
        from model.CPN.encoder import ResnetEncoder
        self.encoder = ResnetEncoder(num_layers=18)

        # -----------------------
        # CPN Feature Mask UNet
        # -----------------------
        from model.CPN.unet import FeatMaskNetwork
        self.fmn = FeatMaskNetwork()

        # -----------------------
        # PAN
        # -----------------------
        from model.PAN import DepthPredictionNetwork
        self.dpn = DepthPredictionNetwork(
            disp_range=disp_range,
            n_planes=num_planes,
        )

        # -----------------------
        # CPN Decoder
        # -----------------------
        from model.CPN.decoder import DepthDecoder
        num_ch_enc = self.encoder.num_ch_enc
        self.decoder = DepthDecoder(
            num_ch_enc=num_ch_enc,
            use_alpha=False,
            scales=range(4),
            use_skips=True,
        )

    def forward(
        self, 
        src_imgs, 
        src_depths, 
        dpn_input_disparity=None,
        dpn_fmn_pretrain=None,
        fix_dpn=None,
        is_train=False,
    ):
        if is_train:
            return self.forward_train(
                src_imgs, src_depths, dpn_input_disparity, dpn_fmn_pretrain, fix_dpn,
            )
        
        rgb_low_res = F.interpolate(src_imgs, size=self.low_res_size, mode='bilinear', align_corners=True)
        disp_low_res = F.interpolate(src_depths, size=self.low_res_size, mode='bilinear', align_corners=True)
        
        bs = src_imgs.shape[0]
        dpn_input_disparity = torch.linspace(
            self.near, 
            self.far, 
            self.num_planes + 2
        )[1:-1].to(src_imgs.device).unsqueeze(0).repeat(bs, 1)
        
        render_disp = self.dpn(dpn_input_disparity, rgb_low_res, disp_low_res)
        feature_mask = self.fmn(src_imgs, src_depths, render_disp)
        # Encoder forward
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.encoder(src_imgs, src_depths)
        enc_features = [conv1_out, block1_out, block2_out, block3_out, block4_out]
        # Decoder forward
        outputs = self.decoder(enc_features, feature_mask)
        return outputs[0], render_disp
    
    def forward_train(
        self,
        src_imgs,
        src_depths,
        dpn_input_disparity,
        dpn_fmn_pretrain,
        fix_dpn,
    ):
        rgb_low_res = F.interpolate(src_imgs, size=self.low_res_size, mode='bilinear', align_corners=True)
        disp_low_res = F.interpolate(src_depths, size=self.low_res_size, mode='bilinear', align_corners=True)

        if not fix_dpn:
            render_disp = self.dpn(dpn_input_disparity, rgb_low_res, disp_low_res)
        else:
            with torch.no_grad():
                render_disp = self.dpn(dpn_input_disparity, rgb_low_res, disp_low_res)

        feature_mask = self.fmn(src_imgs, src_depths, render_disp)

        if dpn_fmn_pretrain:
            # [b,1,h,w] -> [b,s,4,h,w]
            pseudo_outputs = [0.5 * torch.ones_like(src_depths).unsqueeze(1).repeat(1, render_disp.shape[1], 4, 1, 1)]
            return pseudo_outputs, render_disp, feature_mask

        # Encoder forward
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.encoder(src_imgs, src_depths)
        enc_features = [conv1_out, block1_out, block2_out, block3_out, block4_out]
        
        # Decoder forward
        outputs = self.decoder(enc_features, feature_mask)

        return outputs, render_disp, feature_mask
