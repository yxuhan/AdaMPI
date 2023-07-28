import json
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.mpi import mpi_rendering
from utils.mpi.homography_sampler import HomographySample
from utils.utils import AverageMeter, restore_model
from model.utils import edge_aware_loss_v2, masked_psnr, psnr, SSIM
from model.losses import VGGPerceptualLoss, rank_loss_fn, assign_loss


def crop5(img):
    # img: [b,c,h,w] c=1 or 3
    _, _, h, w = img.size()
    hb, wb = h // 20, w // 20
    return img[:, :, hb:h-hb, wb:w-wb]


def _uniform_strategy_fn(near, far, num, device, bs):
    '''
    e.g. near=1, far=0, num=4, return [0.8, 0.6, 0.4, 0.2]
    e.g. near=1, far=0, num=3, return [0.75, 0.50, 0.25]
    '''
    return torch.linspace(near, far, num + 2)[1:-1].to(device).unsqueeze(0).repeat(bs, 1)


class SynthesisTask(nn.Module):
    def __init__(self, rank, config, logger=None, is_val=False):
        super().__init__()
        self.rank = rank
        self.device = torch.device("cuda:%d" % rank)

        # -----------------------
        # MPI Predictor Network All in One
        # -----------------------
        from model.AdaMPI import MPIPredictor
        self.mpi_predict_network = MPIPredictor(
            width=config["data.img_w"],
            height=config["data.img_h"],
            num_planes=config["mpi.num_bins_coarse"],
        )

        # to device
        self.mpi_predict_network = self.mpi_predict_network.to(self.device)
        
        # Init optimizer
        params = [
            {"params": self.mpi_predict_network.parameters(), "lr": config["lr.backbone_lr"]},
        ]
        self.optimizer = torch.optim.Adam(params, weight_decay=config["lr.weight_decay"])

        self.current_epoch = 0
        self.global_step = 0
        self.is_val = is_val

        self.lpips = lpips.LPIPS(net="vgg").cuda(self.rank)
        self.lpips.requires_grad_(False)    
        self.hist = None

        # percept loss
        self.percpt_vgg = VGGPerceptualLoss(resize=False).cuda(self.rank)

        # FFL Loss
        from focal_frequency_loss import FocalFrequencyLoss as FFL
        self.ffl = FFL(loss_weight=1.0, alpha=config.get("training.ffl_alpha", 1))  # initialize nn.Module class

        # Restore model
        model_path = config["training.pretrained_checkpoint_path"]
        if model_path is None:
            print("Not using pre-trained model...")
        else:
            print("Load pre-trained model from", model_path)
            dist.barrier()
            restore_model(model_path, 
                        self.mpi_predict_network,
                        self.optimizer, 
                        logger=None,
                        load_part_list=config.get("training.load_part_list", None))
            dist.barrier()

        # to DDP & train
        self.mpi_predict_network = nn.SyncBatchNorm.convert_sync_batchnorm(self.mpi_predict_network)
        self.mpi_predict_network = DDP(self.mpi_predict_network, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)
        self.mpi_predict_network.train()
        
        # LR scheduling
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            config["lr.decay_steps"],
                                                            gamma=config["lr.decay_gamma"])

        H_tgt, W_tgt = config["data.img_h"], config["data.img_w"]
        self.homography_sampler_list = \
            [HomographySample(H_tgt, W_tgt, device=self.device),
             HomographySample(int(H_tgt / 2), int(W_tgt / 2), device=self.device),
             HomographySample(int(H_tgt / 4), int(W_tgt / 4), device=self.device),
             HomographySample(int(H_tgt / 8), int(W_tgt / 8), device=self.device)]
        self.upsample_list = \
            [nn.Identity(),
             nn.Upsample(size=(int(H_tgt / 2), int(W_tgt / 2))),
             nn.Upsample(size=(int(H_tgt / 4), int(W_tgt / 4))),
             nn.Upsample(size=(int(H_tgt / 8), int(W_tgt / 8)))]

        self.ssim = SSIM(size_average=True).to(self.device)

        self.config = config
        self.tb_writer = config.get("tb_writer", None)

        # Keep track of training / validation losses
        self.train_losses = {
            "loss": AverageMeter("train_loss"),
            "loss_rgb_src": AverageMeter("train_loss_rgb_src"),
            "loss_ssim_src": AverageMeter("train_loss_ssim_src"),
            "loss_rgb_tgt": AverageMeter("train_loss_rgb_tgt"),
            "loss_ssim_tgt": AverageMeter("train_loss_ssim_tgt"),
            "loss_occ_ssim_tgt": AverageMeter("train_loss_occ_ssim_tgt"),
            "loss_occ_l1_tgt": AverageMeter("train_loss_occ_l1_tgt"),
            "loss_disp_tgt": AverageMeter("train_disp_loss"),
            "loss_occ_disp_tgt": AverageMeter("train_occ_disp_loss"),
            "loss_ffl_tgt": AverageMeter("train_ffl_loss"),
            "lpips_tgt": AverageMeter("train_lpips_tgt"),
            "psnr_tgt": AverageMeter("train_psnr_tgt"),
            "occ_psnr_tgt": AverageMeter("occ_train_psnr_tgt"),
            "loss_rank": AverageMeter("train_rank_loss"),
            "loss_assign": AverageMeter("train_assign_loss"),
            "loss_percept": AverageMeter("train_percept_loss"),
        }
        self.val_losses = {
            "loss_rgb_src": AverageMeter("val_loss_rgb_src"),
            "loss_ssim_src": AverageMeter("val_loss_ssim_src"),
            "loss_rgb_tgt": AverageMeter("val_loss_rgb_tgt"),
            "loss_ssim_tgt": AverageMeter("val_loss_ssim_tgt"),
            "loss_occ_ssim_tgt": AverageMeter("val_loss_occ_ssim_tgt"),
            "loss_occ_l1_tgt": AverageMeter("val_loss_occ_l1_tgt"),
            "loss_disp_tgt": AverageMeter("val_disp_loss"),
            "loss_occ_disp_tgt": AverageMeter("val_occ_disp_loss"),
            "loss_ffl_tgt": AverageMeter("val_ffl_loss"),
            "lpips_tgt": AverageMeter("val_lpips_tgt"),
            "psnr_tgt": AverageMeter("val_psnr_tgt"),
            "occ_psnr_tgt": AverageMeter("occ_val_psnr_tgt"),
            "loss_rank": AverageMeter("val_rank_loss"),
            "loss_assign": AverageMeter("val_assign_loss"),
            "loss_percept": AverageMeter("val_percept_loss"), 
        }
        self.loss_for_multi_scale = [
            "loss_ssim_tgt", "loss_occ_ssim_tgt", "loss_rgb_tgt", "loss_occ_l1_tgt", "loss_ffl_tgt",
            "loss_disp_tgt", "loss_occ_disp_tgt"
        ]

    def set_data(self, items):
        names, src_items, tgt_items = items

        self.item_names = names
        self.src_imgs = src_items["img"]
        self.src_depths = src_items["depth"]
        self.K_src = src_items["K"]
        self.K_src_inv = src_items["K_inv"]

        self.occ_mask = tgt_items["occ_mask"]
        self.tgt_imgs = tgt_items["img"]
        self.tgt_depths = tgt_items["depth"]
        self.K_tgt = tgt_items["K"]
        self.K_tgt_inv = tgt_items["K_inv"]
        self.G_src_tgt = tgt_items["G_src_tgt"]

        self.G_tgt_src = torch.inverse(self.G_src_tgt)
    
    def get_disparity_strategy(self, mode, plane_num, device, heu_ratio, bs):
        near, far = self.config["mpi.disparity_start"], self.config["mpi.disparity_end"]
        strategy = None
        if mode == 'uniform':
            strategy = _uniform_strategy_fn(near, far, plane_num, device, bs)
        else:
            raise NotImplementedError
        return strategy

    def loss_fcn_per_scale(self, scale,
                           mpi_all_src, disparity_all_src,
                           scale_factor=None,
                           is_val=False):
        src_imgs_scaled = self.upsample_list[scale](self.src_imgs)
        tgt_imgs_scaled = self.upsample_list[scale](self.tgt_imgs)
        tgt_disps_scaled = self.upsample_list[scale](self.tgt_depths)
        occ_mask_scaled = self.upsample_list[scale](self.occ_mask)

        B, _, H_img_scaled, W_img_scaled = src_imgs_scaled.size()

        K_src_scaled = self.K_src / (2 ** scale)
        K_src_scaled[:, 2, 2] = 1
        K_tgt_scaled = self.K_tgt / (2 ** scale)
        K_tgt_scaled[:, 2, 2] = 1
        # TODO: sometimes it returns identity, unless there is CUDA_LAUNCH_BLOCKING=1
        torch.cuda.synchronize()
        K_src_scaled_inv = torch.inverse(K_src_scaled)

        # compute xyz for src and tgt
        # here we need to ensure mpi resolution == image resolution
        assert mpi_all_src.size(3) == H_img_scaled, mpi_all_src.size(4) == W_img_scaled
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid,
            disparity_all_src,
            K_src_scaled_inv
        )

        # compose depth_src
        # here is blend_weights means how much this plane is visible from the camera, BxSx1xHxW
        # e.g, blend_weights = 0 means it is invisible from the camera
        mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
        mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
        src_imgs_syn, src_depth_syn, blend_weights, weights = mpi_rendering.render(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            xyz_src_BS3HW,
            use_alpha=self.config.get("mpi.use_alpha", False),
            is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )
        if self.config.get("training.src_rgb_blending", True):
            mpi_all_rgb_src = blend_weights * src_imgs_scaled.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
        
        src_imgs_syn, src_depth_syn = mpi_rendering.weighted_sum_mpi(
            mpi_all_rgb_src,
            xyz_src_BS3HW,
            weights,
            is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )

        src_disparity_syn = torch.reciprocal(src_depth_syn)
        
        loss_map_src = torch.abs(src_imgs_syn - src_imgs_scaled)
        loss_rgb_src = loss_map_src.mean()

        # Render target view
        render_results = self.render_novel_view(mpi_all_rgb_src, mpi_all_sigma_src,
                                                disparity_all_src, self.G_tgt_src,
                                                K_src_scaled_inv, K_tgt_scaled,
                                                scale=scale,
                                                scale_factor=scale_factor)

        src_disparity_gt = self.src_depths

        if is_val:
            for k in render_results.keys():
                # tgt_imgs_syn: [bs,3,h,w]
                # tgt_disparity_syn: [bs,1,h,w]
                # tgt_mask_syn: [bs,1,h,w]
                render_results[k] = crop5(render_results[k]) 
            tgt_imgs_scaled = crop5(tgt_imgs_scaled)
            tgt_disps_scaled = crop5(tgt_disps_scaled)
            occ_mask_scaled = crop5(occ_mask_scaled)
            src_disparity_syn = crop5(src_disparity_syn)
            src_imgs_syn = crop5(src_imgs_syn)
            src_disparity_gt = crop5(src_disparity_gt)
            src_imgs_scaled = crop5(src_imgs_scaled)

        tgt_imgs_syn = render_results["tgt_imgs_syn"]
        tgt_disparity_syn = render_results["tgt_disparity_syn"]
        tgt_mask_syn = render_results["tgt_mask_syn"]

        # build loss
        # Read lambdas
        with torch.no_grad():
            loss_rgb_src = torch.mean(torch.abs(src_imgs_syn - src_imgs_scaled))
            loss_ssim_src = 1 - self.ssim(src_imgs_syn, src_imgs_scaled)

        # rgb loss at tgt frame
        # some pixels in tgt frame is outside src FoV, here we can detect and ignore those pixels
        occ_mask_norm = torch.sum(occ_mask_scaled, dim=(-2, -1)) + 1e-5
        rgb_tgt_valid_mask = torch.ge(tgt_mask_syn, self.config["mpi.valid_mask_threshold"]).to(torch.float32)
        loss_map_rgb_tgt = torch.abs(tgt_imgs_syn - tgt_imgs_scaled) * rgb_tgt_valid_mask
        loss_rgb_tgt = loss_map_rgb_tgt.mean()

        loss_occ_rgb_tgt = torch.sum(occ_mask_scaled * loss_map_rgb_tgt, dim=(-2, -1)) / occ_mask_norm
        loss_occ_rgb_tgt = loss_occ_rgb_tgt.mean()

        # SSIM (Occ) loss
        loss_map_ssim_tgt = 1 - self.ssim(tgt_imgs_syn, tgt_imgs_scaled, None)
        loss_ssim_tgt = loss_map_ssim_tgt.mean()

        loss_occ_ssim_tgt = torch.sum(occ_mask_scaled * loss_map_ssim_tgt, dim=(-2, -1)) / occ_mask_norm
        loss_occ_ssim_tgt = loss_occ_ssim_tgt.mean()

        # Disp (Occ) loss
        loss_map_disp_tgt = torch.abs(tgt_disps_scaled - tgt_disparity_syn) * rgb_tgt_valid_mask
        loss_disp_tgt = loss_map_disp_tgt.mean()

        loss_occ_disp_tgt = torch.sum(occ_mask_scaled * loss_map_disp_tgt, dim=(-2, -1)) / occ_mask_norm
        loss_occ_disp_tgt = loss_occ_disp_tgt.mean()

        # focal frequency loss
        loss_ffl_tgt = self.ffl(tgt_imgs_syn, tgt_imgs_scaled)

        # disp smooth loss
        loss_smooth_tgt_v2 = edge_aware_loss_v2(tgt_imgs_scaled, tgt_disparity_syn)
        loss_smooth_src_v2 = edge_aware_loss_v2(src_imgs_scaled, src_disparity_syn)

        # LPIPS and PSNR loss (for eval only):
        with torch.no_grad():
            lpips_tgt = self.lpips(2. * tgt_imgs_syn - 1., 2. * tgt_imgs_scaled - 1.).mean() if is_val and scale == 0 else torch.tensor(0.)
            psnr_tgt = psnr(tgt_imgs_syn, tgt_imgs_scaled).mean()
            # only compute occ loss on the original scale, i.e. scale==0 to avoid the downsampled occ_mask equal to zero map
            occ_psnr_tgt = masked_psnr(tgt_imgs_syn, tgt_imgs_scaled, occ_mask_scaled).mean() if scale == 0 else 0

        loss = loss_rgb_tgt + loss_ssim_tgt \
                + loss_disp_tgt * self.config.get("training.disp_loss_weight", 1) \
                + (loss_occ_ssim_tgt + loss_occ_rgb_tgt) * self.config.get("training.occ_color_loss_weight", 5) \
                + loss_occ_disp_tgt * self.config.get("training.occ_disp_loss_weight", 5) \
                + loss_ffl_tgt * self.config.get("training.ffl_loss_weight", 10) \
                + (loss_smooth_src_v2 + loss_smooth_tgt_v2) * self.config.get("training.smooth_loss_weight", 0.01)

        loss_dict = {"loss": loss,
                     "loss_rgb_src": loss_rgb_src,
                     "loss_ssim_src": loss_ssim_src,
                     "loss_rgb_tgt": loss_rgb_tgt,
                     "loss_ssim_tgt": loss_ssim_tgt,
                     "loss_occ_ssim_tgt": loss_occ_ssim_tgt,
                     "loss_occ_l1_tgt": loss_occ_rgb_tgt,
                     "loss_disp_tgt": loss_disp_tgt,
                     "loss_occ_disp_tgt": loss_occ_disp_tgt,
                     "loss_ffl_tgt": loss_ffl_tgt,
                     "lpips_tgt": lpips_tgt,
                     "psnr_tgt": psnr_tgt,
                     "occ_psnr_tgt": occ_psnr_tgt,
                     "loss_smooth_src": loss_smooth_src_v2,
                     "loss_smooth_tgt": loss_smooth_tgt_v2,
                     }

        visualization_dict = {
                "src_disparity_syn": src_disparity_syn,
                "src_disparity_gt": src_disparity_gt,

                "src_imgs_syn": src_imgs_syn,
                "src_imgs_gt": src_imgs_scaled,

                "tgt_disparity_syn": tgt_disparity_syn,                
                "tgt_disparity_gt": tgt_disps_scaled,
                
                "tgt_imgs_syn": tgt_imgs_syn,                
                "tgt_imgs_gt": tgt_imgs_scaled,                
        }

        mpi = {
            "rgb": mpi_all_rgb_src,
            "sigma": mpi_all_sigma_src,
            "disparity": disparity_all_src, 
            "intrinsics_inv": K_src_scaled_inv, 
        }

        return loss_dict, visualization_dict, mpi

    def loss_fcn(self, is_val):
        loss_dict_list, visualization_dict_list, mpi_list = [], [], []

        # Network forward
        endpoints = self.network_forward()

        # compute loss which do not related to scales
        dpn_output_disparity = endpoints["dpn_output_disparity"]
        dpn_input_disparity = endpoints["dpn_input_disparity"]
        rendered_mpi_disp = endpoints["render_disparity"]
        feat_mask = endpoints["feature_mask"]

        loss_assign = assign_loss(feat_mask, rendered_mpi_disp, self.src_depths)
        loss_rank = rank_loss_fn(dpn_output_disparity, start=self.config["mpi.disparity_start"], end=self.config["mpi.disparity_end"])

        # compute loss for all the scales
        scale_list = list(range(len(endpoints["mpi_all_src_list"])))
        for scale in scale_list:
            loss_dict_tmp, visualization_dict_tmp, mpi_tmp = self.loss_fcn_per_scale(
                scale,
                endpoints["mpi_all_src_list"][scale],
                rendered_mpi_disp.clamp(self.config["mpi.disparity_end"], self.config["mpi.disparity_start"]),
                is_val=is_val
            )
            loss_dict_list.append(loss_dict_tmp)
            visualization_dict_list.append(visualization_dict_tmp)
            mpi_list.append(mpi_tmp)

        # merge loss of all the scales
        loss_dict = loss_dict_list[0]
        visualization_dict = visualization_dict_list[0]
        mpi = mpi_list[0]
        for scale in scale_list[1:]:
            if self.config.get("training.use_multi_scale", True):
                for loss_name in self.loss_for_multi_scale:
                    loss_dict["loss"] += loss_dict_list[scale][loss_name]
        
        # rank loss
        loss_dict["loss"] += loss_rank * self.config.get("training.rank_loss_weight", 1)
        loss_dict["loss_rank"] = loss_rank

        # assign loss
        loss_dict["loss"] += loss_assign * self.config.get("training.assign_loss_weight", 1)
        loss_dict["loss_assign"] = loss_assign

        # --------------------
        # Percept. Loss
        # --------------------
        if not self.is_val:
            loss_percept = self.percpt_vgg(visualization_dict["tgt_imgs_syn"], self.tgt_imgs)
        else:
            loss_percept = torch.zeros_like(loss_rank)
        
        loss_dict["loss_percept"] = loss_percept
        loss_dict["loss"] += loss_percept * self.config.get("training.percept_loss_weight", 0.05)

    
        # visualize origin depth vs. adjusted depth
        visualization_dict["new"] = dpn_output_disparity  # [B,S]
        visualization_dict["old"] = dpn_input_disparity  # [B,S]

        return loss_dict, visualization_dict, mpi

    def network_forward(self):
        # configurations
        bs = self.src_imgs.size(0)
        device = self.src_imgs.device

        # set the number of MPI planes
        plane_num = self.config["mpi.num_bins_coarse"]

        # the input disp strategy to *DPN*
        dpn_input_disparity = self.get_disparity_strategy(
            mode=self.config["disp.dpn_input_strategy"],
            plane_num=plane_num,
            device=device,
            bs=bs,
            heu_ratio=0,
        )

        dpn_fmn_pretrain = True if self.global_step < self.config["training.dpn_fmn_iter"] else False
        if self.is_val:
            dpn_fmn_pretrain = False

        if self.config.get("training.dpn_fix_iter") != None:
            min_iter, max_iter = self.config["training.dpn_fix_iter"]
            fix_dpn = True if (min_iter < self.global_step and self.global_step < max_iter) else False
        else:
            fix_dpn = False

        # Extract MPI and adjusted mpi depth from network
        mpi_all_src_list, render_disparity, feat_mask = self.mpi_predict_network(
            self.src_imgs, 
            self.src_depths,
            dpn_input_disparity,
            dpn_fmn_pretrain,
            fix_dpn,
            is_train=True,
        )
        
        return {
            "mpi_all_src_list": mpi_all_src_list,
            "dpn_output_disparity": render_disparity,
            "dpn_input_disparity": dpn_input_disparity,
            "render_disparity": render_disparity,
            "feature_mask": feat_mask,
        }

    def render_novel_view(self, mpi_all_rgb_src, mpi_all_sigma_src,
                          disparity_all_src, G_tgt_src,
                          K_src_inv, K_tgt, scale=0, scale_factor=None):
        # Apply scale factor
        if scale_factor is not None:
            with torch.no_grad():
                G_tgt_src = torch.clone(G_tgt_src)
                G_tgt_src[:, 0:3, 3] = G_tgt_src[:, 0:3, 3] / scale_factor.view(-1, 1)

        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid,
            disparity_all_src,
            K_src_inv
        )

        xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
            xyz_src_BS3HW,
            G_tgt_src
        )

        # Bx1xHxW, Bx3xHxW, Bx1xHxW
        tgt_imgs_syn, tgt_depth_syn, tgt_mask_syn = mpi_rendering.render_tgt_rgb_depth(
            self.homography_sampler_list[scale],
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            xyz_tgt_BS3HW,
            G_tgt_src,
            K_src_inv,
            K_tgt,
            use_alpha=self.config.get("mpi.use_alpha", False),
            is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )
        tgt_disparity_syn = torch.reciprocal(tgt_depth_syn)

        return {
            "tgt_imgs_syn": tgt_imgs_syn,
            "tgt_disparity_syn": tgt_disparity_syn,
            "tgt_mask_syn": tgt_mask_syn
        }

    def run_eval(self, val_data_loader_dict):
        if self.rank == 0:
            print("Start running evaluation on validation set...")
        self.mpi_predict_network.eval()
        self.is_val = True
        with torch.no_grad():
            for dataset_name, val_data_loader in val_data_loader_dict.items():
                if dataset_name != "coco" and self.global_step % (2000 * self.config["training.step_iter"] ) != 0:
                    continue
                if self.rank == 0:
                    print(f"Validation on {dataset_name}")
                vis_dir = os.path.join(self.config['local_workspace'], f'{self.global_step:06d}', dataset_name)
                # clear train losses average meter
                for val_loss_item in self.val_losses.values():
                    val_loss_item.reset()
                
                record_scores = []
                for i_batch, items in enumerate(val_data_loader):
                    if self.rank == 0:
                        print("    Eval progress: {}/{}".format(i_batch + 1, len(val_data_loader)))

                    self.set_data(items)
                    loss_dict, visualization_dict, mpi = self.loss_fcn(is_val=True)
                    loss_dict = {k: v.cpu() for k, v in loss_dict.items()}
                    dist.barrier()
                    for key, loss_value in loss_dict.items():
                        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                        loss_value /= dist.get_world_size()
                    if self.rank == 0:
                        self.log_val(loss_dict)
                    if self.config.get('visualize', True) and self.rank == 0 and i_batch % 1 == 0:
                        self.visualize(visualization_dict, mpi, vis_dir)
                    # record scores. Note this works only for batch_size == 1 and world_size == 1!
                    record_scores.append({
                        "name": self.item_names[0], 
                        "lpips": loss_dict['lpips_tgt'].item(),
                    })
                
                if self.rank == 0:
                    # Save scores
                    with open(os.path.join(vis_dir, "scores.json"), 'w') as fp:
                        json.dump(record_scores, fp)
                    # log evaluation result
                    print("Evaluation on %s finished, average losses: " % dataset_name)
                    for v in self.val_losses.values():
                        print("    {}".format(v))
                    # Write val losses to tensorboard
                    if self.tb_writer is not None:
                        for key, value in self.val_losses.items():
                            self.tb_writer.add_scalar(key + "/val-" + dataset_name, value.avg, self.global_step)
        self.is_val = False
        self.mpi_predict_network.train()
    
    def save_batch_image(self, imgs, dir_name, img_names):
        '''
        imgs: [b,3,h,w]
        '''
        b = imgs.shape[0]
        for i in range(b):
            save_image(imgs[i], os.path.join(dir_name, "%s.png" % img_names[i]))

    def run_eval_only_image(self, val_data_loader_dict):
        if self.rank == 0:
            print("Start running evaluation on validation set...")
        self.mpi_predict_network.eval()
        self.is_val = True
        with torch.no_grad():
            for dataset_name, val_data_loader in val_data_loader_dict.items():
                if self.rank == 0:
                    print(f"Validation on {dataset_name}")
                vis_dir = os.path.join(self.config['local_workspace'], dataset_name)
                os.makedirs(vis_dir, exist_ok=True)
                src_dir = os.path.join(vis_dir, "src")
                tgt_dir = os.path.join(vis_dir, "tgt")
                render_dir = os.path.join(vis_dir, "render")
                os.makedirs(src_dir, exist_ok=True)
                os.makedirs(tgt_dir, exist_ok=True)
                os.makedirs(render_dir, exist_ok=True)

                # clear train losses average meter
                for val_loss_item in self.val_losses.values():
                    val_loss_item.reset()
                
                record_scores = []
                for i_batch, items in enumerate(val_data_loader):
                    if self.rank == 0:
                        print("    Eval progress: {}/{}".format(i_batch + 1, len(val_data_loader)))

                    self.set_data(items)
                    loss_dict, visualization_dict, mpi = self.loss_fcn(is_val=True)
                    
                    render = visualization_dict["tgt_imgs_syn"]
                    self.save_batch_image(render, render_dir, self.item_names)  # crop 5
                    self.save_batch_image(self.src_imgs, src_dir, self.item_names)
                    self.save_batch_image(self.tgt_imgs, tgt_dir, self.item_names)

                if self.rank == 0:
                    # log evaluation result
                    print("Evaluation on %s finished, average losses: " % dataset_name)
                    for v in self.val_losses.values():
                        print("    {}".format(v))

        self.is_val = False
        self.mpi_predict_network.train()
    
    def log_val(self, loss_dict):
        B = self.src_imgs.size(0)
        # loss logging
        for key, value in self.val_losses.items():
            value.update(loss_dict[key].item(), n=B)

    def visualize(self, visualization_dict, mpi, vis_dir: str):
        B = self.src_imgs.size(0)
        # visualization
        os.makedirs(vis_dir, exist_ok=True)
        vis_name = '_'.join(self.item_names)
        # gt and rendered image / disparity
        vis_list = []
        for k, v in visualization_dict.items():
            if k in ["old", "new"]:
                continue
            vis_list.append(v if 'disp' not in k else v.repeat(1, 3, 1, 1))
        vis = torch.cat(vis_list)
        vis_path = os.path.join(vis_dir, vis_name + '.png')
        save_image(vis, vis_path, nrow=B)
        
        # visualize mpi disparity before and after DPN
        old_disparity = visualization_dict["old"].cpu()
        new_disparity = visualization_dict["new"].cpu()
        y_disparity = torch.zeros_like(old_disparity[0])

        src_depth = self.src_depths.cpu().numpy()
        near, far = self.config["mpi.disparity_start"], self.config["mpi.disparity_end"]
        for i in range(B):
            hist_depth, _ = np.histogram(src_depth[i], bins=256, range=(far, near), density=True)
            hist_x = torch.linspace(far, near, 256).numpy()
            plt.subplot(B, 1, i + 1)
            plt.plot(hist_x, hist_depth)
            if self.hist is not None:
                plt.plot(hist_x, self.hist[i])
            plt.plot(old_disparity[i], y_disparity, '.', color='r')
            plt.plot(new_disparity[i], y_disparity, 'x', color='g')   
        plt.savefig(os.path.join(vis_dir, vis_name + "_dpn.jpg"))
        plt.clf()

    def log_training(self, epoch, step, global_step, dataset_length, loss_dict):
        loss = loss_dict["loss"]
        loss_rgb_tgt = loss_dict["loss_rgb_tgt"]
        loss_ssim_tgt = loss_dict["loss_ssim_tgt"]
        loss_rgb_src = loss_dict["loss_rgb_src"]
        loss_ssim_src = loss_dict["loss_ssim_src"]
        
        print(
            "epoch [%.3d] step [%d/%d] global_step = %d total_loss = %.4f encoder_lr = %.7f\n"
            "        src: rgb = %.4f\n"
            "        src: ssim = %.4f\n"
            "        tgt: rgb = %.4f\n"
            "        tgt: ssim = %.4f\n" %
            (epoch, step, dataset_length, self.global_step,
             loss.item(), self.optimizer.param_groups[0]["lr"],
             loss_rgb_src.item(),
             loss_ssim_src.item(),
             loss_rgb_tgt.item(),
             loss_ssim_tgt.item())
        )

        # Write losses to tensorboard
        # Update avg meters
        for key, value in self.train_losses.items():
            self.tb_writer.add_scalar(key + "/train", loss_dict[key].item(), global_step)
            value.update(loss_dict[key].item())

    def train_epoch(self, train_data_loader, val_data_loader_dict, epoch):
        
        if hasattr(train_data_loader, "sampler"):
            train_data_loader.sampler.set_epoch(epoch)

        self.mpi_predict_network.train()

        self.current_epoch = epoch
        self.config["current_epoch"] = epoch

        # clear train losses average meter
        for train_loss_item in self.train_losses.values():
            train_loss_item.reset()

        self.optimizer.zero_grad()
        # iterate over the dataloader
        for step, items in enumerate(train_data_loader):
            step += 1

            self.global_step += 1
            self.set_data(items)

            loss_dict, _, _ = self.loss_fcn(is_val=False)
            loss = loss_dict["loss"] / self.config["training.step_iter"]

            loss.backward()
            if self.global_step % self.config["training.step_iter"] == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # logging
            if step > 0 and step % 10 == 0 and self.rank == 0:
                self.log_training(self.current_epoch,
                                  step,
                                  self.global_step,
                                  len(train_data_loader),
                                  loss_dict)

            if self.rank == 0 and self.global_step > 0 and (self.global_step == 2000 or (self.global_step % 2000 == 0)):
                # Save model and put checkpoint to hdfs
                checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_%06d.pth" % self.global_step)
                torch.save({"mpi_predict_network": self.mpi_predict_network.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "global_step": self.global_step,
                            "current_epoch": self.current_epoch},
                           checkpoint_path)

            if self.global_step == 50 or (self.global_step % self.config["training.eval_interval"] == 0):   
                self.run_eval(val_data_loader_dict)
            
    def train(self, train_data_loader, val_data_loader_dict):
        for epoch in range(1, self.config["training.epochs"] + 1):
            self.current_epoch = epoch
            self.train_epoch(train_data_loader, val_data_loader_dict, epoch)

            if self.rank == 0:
                print("Epoch finished, average losses: ")
                for v in self.train_losses.values():
                    print("    {}".format(v))
