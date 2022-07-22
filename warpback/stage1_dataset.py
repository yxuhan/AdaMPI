import sys
sys.path.append(".")
sys.path.append("..")
import os
import glob
import math
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.utils import save_image

from warpback.utils import (
    RGBDRenderer, 
    image_to_tensor, 
    disparity_to_tensor,
    transformation_from_parameters,
)


class WarpBackStage1Dataset(Dataset):
    def __init__(
        self,
        data_root,
        width=384,
        height=256,
        depth_dir_name="dpt_depth",
        device="cuda",  # device of mesh renderer
        trans_range={"x":0.2, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1},  # xyz for translation, abc for euler angle
    ):
        self.data_root = data_root
        self.depth_dir_name = depth_dir_name
        self.renderer = RGBDRenderer(device)
        self.width = width
        self.height = height
        self.device = device
        self.trans_range = trans_range
        self.image_path_list = glob.glob(os.path.join(self.data_root, "*.jpg"))
        self.image_path_list += glob.glob(os.path.join(self.data_root, "*.png"))

        # set intrinsics
        self.K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
        ]).to(device)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        disp_path = os.path.join(self.data_root, self.depth_dir_name, "%s.png" % image_name)
        
        image = image_to_tensor(image_path, unsqueeze=False)  # [3,h,w]
        disp = disparity_to_tensor(disp_path, unsqueeze=False)  # [1,h,w]
        
        # do some data augmentation, ensure the rgbd spatial resolution is (self.height, self.width)
        image, disp = self.preprocess_rgbd(image, disp)
        
        return image, disp
    
    def preprocess_rgbd(self, image, disp):
        # NOTE 
        # (1) here we directly resize the image to the target size (self.height, self.width)
        # a better way is to first crop a random patch from the image according to the height-width ratio
        # then resize this patch to the target size
        # (2) another suggestion is, add some code to filter the depth map to reduce artifacts around 
        # depth discontinuities
        image = F.interpolate(image.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        disp = F.interpolate(disp.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        return image, disp
    
    def get_rand_ext(self, bs):
        x, y, z = self.trans_range['x'], self.trans_range['y'], self.trans_range['z']
        a, b, c = self.trans_range['a'], self.trans_range['b'], self.trans_range['c']
        cix = self.rand_tensor(x, bs)
        ciy = self.rand_tensor(y, bs)
        ciz = self.rand_tensor(z, bs)
        aix = self.rand_tensor(math.pi / a, bs)
        aiy = self.rand_tensor(math.pi / b, bs)
        aiz = self.rand_tensor(math.pi / c, bs)
        
        axisangle = torch.cat([aix, aiy, aiz], dim=-1)  # [b,1,3]
        translation = torch.cat([cix, ciy, ciz], dim=-1)
        
        cam_ext = transformation_from_parameters(axisangle, translation)  # [b,4,4]
        cam_ext_inv = torch.inverse(cam_ext)  # [b,4,4]
        return cam_ext[:, :-1], cam_ext_inv[:, :-1]
    
    def rand_tensor(self, r, l):
        '''
        return a tensor of size [l], where each element is in range [-r,-r/2] or [r/2,r]
        '''
        if r < 0:  # we can set a negtive value in self.trans_range to avoid random transformation
            return torch.zeros((l, 1, 1))
        rand = torch.rand((l, 1, 1))        
        sign = 2 * (torch.randn_like(rand) > 0).float() - 1
        return sign * (r / 2 + r / 2 * rand)

    def collect_data(self, batch):
        batch = default_collate(batch)
        image, disp = batch
        image = image.to(self.device)
        disp = disp.to(self.device)
        rgbd = torch.cat([image, disp], dim=1)  # [b,4,h,w]
        b = image.shape[0]

        cam_int = self.K.repeat(b, 1, 1)  # [b,3,3]
        cam_ext, cam_ext_inv = self.get_rand_ext(b)  # [b,3,4]
        cam_ext = cam_ext.to(self.device)
        cam_ext_inv = cam_ext_inv.to(self.device)
        
        # warp to a random novel view
        mesh = self.renderer.construct_mesh(rgbd, cam_int)
        warp_image, warp_disp, warp_mask = self.renderer.render_mesh(mesh, cam_int, cam_ext)
        
        # warp back to the original view
        warp_rgbd = torch.cat([warp_image, warp_disp], dim=1)  # [b,4,h,w]
        warp_mesh = self.renderer.construct_mesh(warp_rgbd, cam_int)
        warp_back_image, warp_back_disp, mask = self.renderer.render_mesh(warp_mesh, cam_int, cam_ext_inv)

        # NOTE
        # (1) to train the inpainting network, you only need image, disp, and mask
        # (2) you can add some morphological operation to refine the mask
        return {
            "rgb": image,
            "disp": disp,
            "mask": mask,
            "warp_rgb": warp_image,
            "warp_disp": warp_disp,
            "warp_back_rgb": warp_back_image,
            "warp_back_disp": warp_back_disp,
        }
    

if __name__ == "__main__":
    bs = 8
    data = WarpBackStage1Dataset(
        data_root="warpback/toydata",
    )
    loader = DataLoader(
        dataset=data,
        batch_size=bs,
        shuffle=True,
        collate_fn=data.collect_data,
    )
    for idx, batch in enumerate(loader):
        image, disp, mask = batch["rgb"], batch["disp"], batch["mask"]
        w_image, w_disp = batch["warp_rgb"], batch["warp_disp"]
        wb_image, wb_disp = batch["warp_back_rgb"], batch["warp_back_disp"]
        visual = torch.cat([
            image,
            disp.repeat(1, 3, 1, 1),
            mask.repeat(1, 3, 1, 1),
            wb_image,
            wb_disp.repeat(1, 3, 1, 1),
            w_image,
            w_disp.repeat(1, 3, 1, 1),
        ], dim=0)
        save_image(visual, "debug/stage1-%03d.jpg" % idx, nrow=bs)
