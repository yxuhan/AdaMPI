import sys
sys.path.append(".")
sys.path.append("..")
import os
import glob
import math
import numpy as np
from skimage.feature import canny
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.utils import save_image
from torchvision import transforms

from warpback.utils import (
    RGBDRenderer, 
    image_to_tensor, 
    disparity_to_tensor,
    transformation_from_parameters,
)
from warpback.networks import get_edge_connect


class WarpBackStage2Dataset(Dataset):
    def __init__(
        self,
        data_root,
        width=384,
        height=256,
        depth_dir_name="dpt_depth",
        device="cuda",  # device of mesh renderer
        trans_range={"x":0.2, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1},  # xyz for translation, abc for euler angle
        ec_weight_dir="warpback/ecweight",
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

        # get Stage-1 pretrained inpainting network
        self.edge_model, self.inpaint_model, self.disp_model = get_edge_connect(ec_weight_dir)
        self.edge_model = self.edge_model.to(self.device)
        self.inpaint_model = self.inpaint_model.to(self.device)
        self.disp_model = self.disp_model.to(self.device)

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

    def inpaint(self, image, disp, mask):
        image_gray = transforms.Grayscale()(image)
        edge = self.get_edge(image_gray, mask)
        
        mask_hole = 1 - mask

        # inpaint edge
        edge_model_input = torch.cat([image_gray, edge, mask_hole], dim=1)  # [b,4,h,w]
        edge_inpaint = self.edge_model(edge_model_input)  # [b,1,h,w]

        # inpaint RGB
        inpaint_model_input = torch.cat([image + mask_hole, edge_inpaint], dim=1)
        image_inpaint = self.inpaint_model(inpaint_model_input)
        image_merged = image * (1 - mask_hole) + image_inpaint * mask_hole
        
        # inpaint Disparity
        disp_model_input = torch.cat([disp + mask_hole, edge_inpaint], dim=1)
        disp_inpaint = self.disp_model(disp_model_input)
        disp_merged = disp * (1 - mask_hole) + disp_inpaint * mask_hole

        return image_merged, disp_merged

    def get_edge(self, image_gray, mask):
        image_gray_np = image_gray.squeeze(1).cpu().numpy()  # [b,h,w]
        mask_bool_np = np.array(mask.squeeze(1).cpu(), dtype=np.bool_)  # [b,h,w]
        edges = []
        for i in range(mask.shape[0]):
            cur_edge = canny(image_gray_np[i], sigma=2, mask=mask_bool_np[i])
            edges.append(torch.from_numpy(cur_edge).unsqueeze(0))  # [1,h,w]
        edge = torch.cat(edges, dim=0).unsqueeze(1).float()  # [b,1,h,w]
        return edge.to(self.device)

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
        
        # warp to a random novel view and inpaint the holes
        # as the source view (input view) to the single-view view synthesis method
        mesh = self.renderer.construct_mesh(rgbd, cam_int)
        warp_image, warp_disp, warp_mask = self.renderer.render_mesh(mesh, cam_int, cam_ext)
        
        with torch.no_grad():
            src_image, src_disp = self.inpaint(warp_image, warp_disp, warp_mask)

        return {
            "src_rgb": src_image,
            "src_disp": src_disp,
            "tgt_rgb": image,
            "tgt_disp": disp,
            "warp_rgb": warp_image,
            "warp_disp": warp_disp,
            "cam_int": cam_int,  # src and tgt view share the same intrinsic
            "cam_ext": cam_ext_inv,
        }
    

if __name__ == "__main__":
    bs = 8
    data = WarpBackStage2Dataset(
        data_root="warpback/toydata",
    )
    loader = DataLoader(
        dataset=data,
        batch_size=bs,
        shuffle=True,
        collate_fn=data.collect_data,
    )
    for idx, batch in enumerate(loader):
        src_rgb, src_disp = batch["src_rgb"], batch["src_disp"]
        tgt_rgb, tgt_disp = batch["tgt_rgb"], batch["tgt_disp"]
        warp_rgb, warp_disp = batch["warp_rgb"], batch["warp_disp"]
        visual = torch.cat([
            warp_rgb,
            warp_disp.repeat(1, 3, 1, 1),
            src_rgb,
            src_disp.repeat(1, 3, 1, 1),
            tgt_rgb,
            tgt_disp.repeat(1, 3, 1, 1),
        ], dim=0)
        save_image(visual, "debug/stage2-%03d.jpg" % idx, nrow=bs)
