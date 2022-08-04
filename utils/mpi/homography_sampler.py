import torch
import numpy as np
from scipy.spatial.transform import Rotation


def inverse(matrices):
    """
    torch.inverse() sometimes produces outputs with nan the when batch size is 2.
    Ref https://github.com/pytorch/pytorch/issues/47272
    this function keeps inversing the matrix until successful or maximum tries is reached
    :param matrices Bx3x3
    """
    inverse = None
    max_tries = 5
    while (inverse is None) or (torch.isnan(inverse)).any():
        torch.cuda.synchronize()
        inverse = torch.inverse(matrices)

        # Break out of the loop when the inverse is successful or there"re no more tries
        max_tries -= 1
        if max_tries == 0:
            break

    # Raise an Exception if the inverse contains nan
    if (torch.isnan(inverse)).any():
        raise Exception("Matrix inverse contains nan!")
    return inverse


class HomographySample:
    def __init__(self, H_tgt, W_tgt, device=None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.Height_tgt = H_tgt
        self.Width_tgt = W_tgt
        self.meshgrid = self.grid_generation(self.Height_tgt, self.Width_tgt, self.device)
        self.meshgrid = self.meshgrid.permute(2, 0, 1).contiguous()  # 3xHxW

        self.n = self.plane_normal_generation(self.device)

    @staticmethod
    def grid_generation(H, W, device):
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xv, yv = np.meshgrid(x, y)  # HxW
        xv = torch.from_numpy(xv.astype(np.float32)).to(dtype=torch.float32, device=device)
        yv = torch.from_numpy(yv.astype(np.float32)).to(dtype=torch.float32, device=device)
        ones = torch.ones_like(xv)
        meshgrid = torch.stack((xv, yv, ones), dim=2)  # HxWx3
        return meshgrid

    @staticmethod
    def plane_normal_generation(device):
        n = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        return n

    @staticmethod
    def euler_to_rotation_matrix(x_angle, y_angle, z_angle, seq='xyz', degrees=False):
        """
        Note that here we want to return a rotation matrix rot_mtx, which transform the tgt points into src frame,
        i.e, rot_mtx * p_tgt = p_src
        Therefore we need to add negative to x/y/z_angle
        :param roll:
        :param pitch:
        :param yaw:
        :return:
        """
        r = Rotation.from_euler(seq,
                                [-x_angle, -y_angle, -z_angle],
                                degrees=degrees)
        rot_mtx = r.as_matrix().astype(np.float32)
        return rot_mtx


    def sample(self, src_BCHW, d_src_B,
               G_tgt_src,
               K_src_inv, K_tgt):
        """
        Coordinate system: x, y are the image directions, z is pointing to depth direction
        :param src_BCHW: torch tensor float, 0-1, rgb/rgba. BxCxHxW
                         Assume to be at position P=[I|0]
        :param d_src_B: distance of image plane to src camera origin
        :param G_tgt_src: Bx4x4
        :param K_src_inv: Bx3x3
        :param K_tgt: Bx3x3
        :return: tgt_BCHW
        """
        # parameter processing ------ begin ------
        B, channels, Height_src, Width_src = src_BCHW.size(0), src_BCHW.size(1), src_BCHW.size(2), src_BCHW.size(3)
        R_tgt_src = G_tgt_src[:, 0:3, 0:3]
        t_tgt_src = G_tgt_src[:, 0:3, 3]

        Height_tgt = self.Height_tgt
        Width_tgt = self.Width_tgt
        # if R_src_tgt is None:
        #     R_src_tgt = torch.eye(3, dtype=torch.float32, device=src_BCHW.device)
        #     R_src_tgt = R_src_tgt.unsqueeze(0).expand(B, 3, 3)
        # if t_src_tgt is None:
        #     t_src_tgt = torch.tensor([0, 0, 0],
        #                              dtype=torch.float32,
        #                              device=src_BCHW.device)
        #     t_src_tgt = t_src_tgt.unsqueeze(0).expand(B, 3)

        # relationship between FoV and focal length:
        # assume W > H
        # W / 2 = f*tan(\theta / 2)
        # here we default the horizontal FoV as 53.13 degree
        # the vertical FoV can be computed as H/2 = W*tan(\theta/2)

        R_tgt_src = R_tgt_src.to(device=src_BCHW.device)
        t_tgt_src = t_tgt_src.to(device=src_BCHW.device)
        K_src_inv = K_src_inv.to(device=src_BCHW.device)
        K_tgt = K_tgt.to(device=src_BCHW.device)
        # parameter processing ------ end ------

        # the goal is compute H_src_tgt, that maps a tgt pixel to src pixel
        # so we compute H_tgt_src first, and then inverse
        n = self.n.to(device=src_BCHW.device)
        n = n.unsqueeze(0).repeat(B, 1)  # Bx3
        # Bx3x3 - (Bx3x1 * Bx1x3)
        # note here we use -d_src, because the plane function is n^T * X - d_src = 0
        d_src_B33 = d_src_B.reshape(B, 1, 1).repeat(1, 3, 3)  # B -> Bx3x3
        R_tnd = R_tgt_src - torch.matmul(t_tgt_src.unsqueeze(2), n.unsqueeze(1)) / -d_src_B33
        H_tgt_src = torch.matmul(K_tgt,
                                 torch.matmul(R_tnd, K_src_inv))

        # TODO: fix cuda inverse
        with torch.no_grad():
            H_src_tgt = inverse(H_tgt_src)

        # create tgt image grid, and map to src
        meshgrid_tgt_homo = self.meshgrid.to(src_BCHW.device)
        # 3xHxW -> Bx3xHxW
        meshgrid_tgt_homo = meshgrid_tgt_homo.unsqueeze(0).expand(B, 3, Height_tgt, Width_tgt)

        # wrap meshgrid_tgt_homo to meshgrid_src
        meshgrid_tgt_homo_B3N = meshgrid_tgt_homo.view(B, 3, -1)  # Bx3xHW
        meshgrid_src_homo_B3N = torch.matmul(H_src_tgt, meshgrid_tgt_homo_B3N)  # Bx3x3 * Bx3xHW -> Bx3xHW
        # Bx3xHW -> Bx3xHxW -> BxHxWx3
        meshgrid_src_homo = meshgrid_src_homo_B3N.view(B, 3, Height_tgt, Width_tgt).permute(0, 2, 3, 1)
        meshgrid_src = meshgrid_src_homo[:, :, :, 0:2] / meshgrid_src_homo[:, :, :, 2:]  # BxHxWx2

        valid_mask_x = torch.logical_and(meshgrid_src[:, :, :, 0] < Width_src,
                                         meshgrid_src[:, :, :, 0] > -1)
        valid_mask_y = torch.logical_and(meshgrid_src[:, :, :, 1] < Height_src,
                                         meshgrid_src[:, :, :, 1] > -1)
        valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # BxHxW

        # sample from src_BCHW
        # normalize meshgrid_src to [-1,1]
        meshgrid_src[:, :, :, 0] = (meshgrid_src[:, :, :, 0]+0.5) / (Width_src * 0.5) - 1
        meshgrid_src[:, :, :, 1] = (meshgrid_src[:, :, :, 1]+0.5) / (Height_src * 0.5) - 1
        tgt_BCHW = torch.nn.functional.grid_sample(src_BCHW, grid=meshgrid_src, padding_mode='border',
                                                   align_corners=False)
        # BxCxHxW, BxHxW
        return tgt_BCHW, valid_mask
