import os
import math
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from moviepy.editor import ImageSequenceClip

from utils.mpi import mpi_rendering
from utils.mpi.homography_sampler import HomographySample


def image_to_tensor(img_path, unsqueeze=True):
    rgb = transforms.ToTensor()(Image.open(img_path))
    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb


def disparity_to_tensor(disp_path, unsqueeze=True):
    disp = cv2.imread(disp_path, -1) / (2 ** 16 - 1)
    disp = torch.from_numpy(disp)[None, ...]
    if unsqueeze:
        disp = disp.unsqueeze(0)
    return disp.float()


def gen_swing_path(num_frames=90, r_x=0.14, r_y=0., r_z=0.10):
    "Return a list of matrix [4, 4]"
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
    return poses.unbind()


def render_3dphoto(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src

    # render novel views
    swing_path_list = gen_swing_path()
    frames = []
    for cam_ext in tqdm(swing_path_list):
        frame = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            homography_sampler,
        )
        frame_np = frame[0].permute(1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255), a_min=0, a_max=255).astype(np.uint8)
        frames.append(frame_np)
    rgb_clip = ImageSequenceClip(frames, fps=30)
    rgb_clip.write_videofile(save_path, verbose=False, codec='mpeg4', logger=None, bitrate='2000k')


def render_novel_view(
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    homography_sampler,
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW,
        G_tgt_src
    )

    tgt_imgs_syn, _, _ = mpi_rendering.render_tgt_rgb_depth(
        homography_sampler,
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        xyz_tgt_BS3HW,
        G_tgt_src,
        K_src_inv,
        K_tgt,
        use_alpha=False,
        is_bg_depth_inf=False,
    )

    return tgt_imgs_syn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        # return fmtstr.format(**self.__dict__)
        return f"{self.name:s}: {self.avg:.6f}"
