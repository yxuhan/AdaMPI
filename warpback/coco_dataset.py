import sys
sys.path.append('.')
sys.path.append('..')

import os
import random
import math
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
from skimage.filters import sobel
from skimage.feature import canny
from scipy.interpolate import griddata
import cv2
import pickle
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader

from .utils import seamless_clone, transformation_from_parameters, RGBDRenderer
from .networks import get_edge_connect


class COCODataset(data.Dataset):
    def __init__(
        self, 
        data_root,
        depth_root,
        height, 
        width, 
        keep_aspect_ratio=True, 
        debug=False, 
        training=True, 
        sigma=2,
        ec_weight_dir=None, 
        rand_trans=True, 
        trans_range=None, 
        rank=0, 
        trans_sign=[-1, 1], 
        gt_as_src=False,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.data_root = data_root
        self.depth_root = depth_root
        self.trans_sign = trans_sign
        self.gt_as_src = gt_as_src
        self.feed_height = height
        self.feed_width = width
        self.training = training
        self.debug = debug
        self.sigma = sigma  # for canny edge detector
        self.keep_aspect_ratio = keep_aspect_ratio
        self.rand_trans = rand_trans
        self.trans_range = trans_range
        self.file_names = self.get_file_names()
        self.xs, self.ys = np.meshgrid(np.arange(self.feed_width), np.arange(self.feed_height))
        self.gray_tensor_transform = transforms.Grayscale()

        # load pretrained edge-connect model
        self.edge_model, self.inpaint_model, self.disp_model = get_edge_connect(ec_weight_dir)
        self.edge_model = self.edge_model.cuda(rank)
        self.inpaint_model = self.inpaint_model.cuda(rank)
        self.disp_model = self.disp_model.cuda(rank)        

        # set intrinsics
        self.K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
        ])
        self.K_normalized = torch.clone(self.K)
        self.K[0, :] *= self.feed_width
        self.K[1, :] *= self.feed_height
        self.inv_K = torch.inverse(self.K)
        self.K_row = torch.tensor([[0.58, 0.58, 0.5, 0.5]]).cuda(self.rank)

        self.renderer = RGBDRenderer(device=self.rank)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(self.file_names[index])
        except:
            item = self.load_item(self.file_names[0])
        return item

    def get_file_names(self):
        image_file_names_cache = os.path.join(self.data_root, 'file_names.pkl')
        if os.path.exists(image_file_names_cache):
            with open(image_file_names_cache, 'rb') as fp:
                image_file_names = pickle.load(fp)
        else:
            image_file_names = os.listdir(self.data_root)
            with open(image_file_names_cache, 'wb') as fp:
                pickle.dump(image_file_names, fp, protocol=pickle.HIGHEST_PROTOCOL)

        depth_file_names_cache = os.path.join(self.depth_root, 'file_names.pkl')
        if os.path.exists(depth_file_names_cache):
            with open(depth_file_names_cache, 'rb') as fp:
                depth_file_names = pickle.load(fp)
        else:
            depth_file_names = os.listdir(self.depth_root)
            with open(depth_file_names_cache, 'wb') as fp:
                pickle.dump(depth_file_names, fp, protocol=pickle.HIGHEST_PROTOCOL)

        file_names = sorted(list(set(s[:-4] for s in image_file_names).intersection(set(s[:-4] for s in depth_file_names))))
        return file_names
        
    def load_item(self, file_name):
        # load image and the corresponding disparity
        image = self.load_image(os.path.join(self.data_root, file_name + '.jpg'))
        disparity = self.load_disparity(os.path.join(self.depth_root, file_name + '.png'))

        # resize
        image, disparity = self.prepare_sizes(image, disparity)

        # sharpen disparity
        disparity = self.process_disparity(disparity)

        image = torch.from_numpy(image).float()  # [h, w, 3] 
        disparity = torch.from_numpy(disparity).unsqueeze(-1).float()  # [h, w, 1]

        image = image.permute(2, 0, 1)          # [3, h, w]
        disparity = disparity.permute(2, 0, 1)  # [1, h, w]

        tgt_item_uncropped = {
            "img" : image,
            "depth" : disparity,
            "K" : self.K,
            "K_inv" : self.inv_K
        }

        src_item = {
            "K" : self.K,
            "K_inv" : self.inv_K,
        }

        # crop
        tgt_item = self.crop_all(tgt_item_uncropped)

        return file_name, src_item, tgt_item
        
    def load_image(self, img_path):
        with Image.open(img_path) as img:
            img_uint8 = np.array(img.convert('RGB'))
        return img_uint8.astype(np.float32) / 255.

    def load_disparity(self, disp_path):
        with Image.open(disp_path) as img:
            disp_uint16 = np.array(img)
        return disp_uint16.astype(np.float32) / 65535.

    def prepare_sizes(self, image, disparity):
        height, width, _ = image.shape

        if self.keep_aspect_ratio:
            # check the constraint
            current_ratio = height / width
            target_ratio = self.feed_height / self.feed_width
            if current_ratio < target_ratio:
                # height is the constraint
                target_height = self.feed_height
                target_width = int(self.feed_height / height * width)
            elif current_ratio > target_ratio:
                # width is the constraint
                target_height = int(self.feed_width / width * height)
                target_width = self.feed_width
            else:
                # ratio is the same - just resize
                target_height = self.feed_height
                target_width = self.feed_width
        else:
            target_height = self.feed_height
            target_width = self.feed_width

        image = cv2.resize(image, (target_width, target_height))
        disparity = cv2.resize(disparity, (target_width, target_height))
        return image, disparity

    def crop_all(self, inputs):
        # get crop parameters
        height, width = inputs['img'].shape[-2:]
        if self.gt_as_src:
            top = int(0.5 * (height - self.feed_height))
            left = int(0.5 * (width - self.feed_width))
        else:
            top = int(random.random() * (height - self.feed_height))
            left = int(random.random() * (width - self.feed_width))
        right, bottom = left + self.feed_width, top + self.feed_height

        inputs['img'] = inputs['img'][..., top:bottom, left:right]
        inputs['depth'] = inputs['depth'][...,  top:bottom, left:right]

        return inputs

    def process_disparity(self, disparity):
        disparity = disparity.copy()

        # make disparities positive
        max_disp = disparity.max()
        min_disp = disparity.min()
        if min_disp < 0:
            disparity += np.abs(min_disp)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
        
        # disparity /= disparity.max()  # now 0-1

        # sharpen disparity
        # now find disparity gradients and set to nearest - stop flying pixels
        edges = sobel(disparity) > 0.03  # bool tensor size和img一样
        s = np.sum(edges)

        mask_ravel = (edges == False).ravel()
        rmask_ravel = edges.ravel()

        xs, ys = np.meshgrid(np.arange(disparity.shape[-1]), np.arange(disparity.shape[-2]))
        xs, ys = xs.ravel(), ys.ravel()

        xy1_mask = np.stack([xs[mask_ravel], ys[mask_ravel]], 1)
        f_mask = disparity.ravel()[mask_ravel]
        xy1_rmask = np.stack([xs[rmask_ravel], ys[rmask_ravel]], 1)

        try:
            disparity.ravel()[rmask_ravel] = griddata(xy1_mask, f_mask, xy1_rmask, method='nearest')
        except (ValueError, IndexError) as e:
            print('[Warning] griddata error because:', e)

        return disparity

    def load_mask(self, rgbd):
        '''
        input:
            rgbd: [b,4,h,w]
        output:
            mask: [b,1,h,w]
            warp_rgb: [b,3,h,w]
            warp_d: [b,1,h,w]
        '''
        bs = rgbd.shape[0]
        input_pose, output_pose = self.get_rand_ext(bs)
        rel_pose = (torch.inverse(input_pose) @ output_pose).to(rgbd)[:, :-1]  # [b,3,4]
        cam_int = self.K_normalized[None, ...].repeat(bs, 1, 1).to(rgbd)  # [b,3,3]
        mesh = self.renderer.construct_mesh(rgbd, cam_int)
        render, disp, mask = self.renderer.render_mesh(mesh, cam_int, rel_pose)
        mask[mask>0.5] = 1
        mask[mask<0.5] = 0
        warp_rgb = render
        warp_d = disp
        return mask, warp_rgb, warp_d, input_pose, output_pose
    
    def get_occ_mask(self, rgbd, input_pose, output_pose):
        bs = rgbd.shape[0]
        rel_pose = (torch.inverse(input_pose) @ output_pose).to(rgbd)[:, :-1]  # [b,3,4]
        cam_int = self.K_normalized[None, ...].repeat(bs, 1, 1).to(rgbd)  # [b,3,3]
        mesh = self.renderer.construct_mesh(rgbd, cam_int)
        render, disp, mask = self.renderer.render_mesh(mesh, cam_int, rel_pose)
        mask[mask>0.5] = 1
        mask[mask<0.5] = 0
        warp_rgb = render
        warp_d = disp
        return 1 - mask, warp_rgb, warp_d

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
        
        input_pose = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]).float().repeat(bs, 1, 1)

        ref_view = transformation_from_parameters(axisangle, translation)
        self.G_src_tgt = ref_view

        output_pose = ref_view
        return input_pose, output_pose

    def rand_tensor(self, r, l):
        '''
        a random tensor with length l, in range [-r,-r/2] or [r/2,r]
        '''
        if r < 0:
            return torch.zeros((l, 1, 1))
        
        if self.rand_trans:        
            rand = torch.rand((l, 1, 1))
        else:
            rand = torch.ones((l, 1, 1))
        
        sign = random.choice(self.trans_sign)
        return sign * (r / 2 + r / 2 * rand)

    def edge_connect_inpaint(self, img, img_gray, disp, edge, mask):
        edge_model_input = torch.cat([img_gray, edge, mask], dim=1)  # [bs, 4, h, w]
        edge_inpaint = self.edge_model(edge_model_input)  # [bs,1,h,w]

        # inpaint RGB
        inpaint_model_input = torch.cat([img + mask, edge_inpaint], dim=1)
        img_inpaint = self.inpaint_model(inpaint_model_input)
        img_merged = seamless_clone(img_inpaint, img, mask)
        
        # inpaint Disparity
        disp_model_input = torch.cat([disp + mask, edge_inpaint], dim=1)
        disp_inpaint = self.disp_model(disp_model_input)
        disp_merged = seamless_clone(disp_inpaint, disp, mask) 

        return img_merged, disp_merged
    
    def load_edge(self, warp_gray, mask):
        warp_gray_np = warp_gray.squeeze(1).cpu().numpy()  # [b, h, w]
        mask_bool_np = np.array(mask.squeeze(1).cpu(), dtype=np.bool_)  # [b, h, w]
        edges = []
        for i in range(mask.shape[0]):
            cur_edge = canny(warp_gray_np[i], sigma=self.sigma, mask=mask_bool_np[i])
            edges.append(torch.from_numpy(cur_edge).unsqueeze(0))  # [1,h,w]
        edge = torch.cat(edges, dim=0).unsqueeze(1).float()  # [b,1,h,w]
        return edge.cuda(self.rank)

    def collate_fn(self, batch):
        names, _src_items, _tgt_items = zip(*batch)
        tgt_items = default_collate(_tgt_items)
        src_items = default_collate(_src_items)

        tgt_items = {key: value.cuda(self.rank) for key, value in tgt_items.items()}
        src_items = {key: value.cuda(self.rank) for key, value in src_items.items()}

        with torch.no_grad():
            if self.training:
                rgbd = torch.cat([tgt_items["img"], tgt_items["depth"]], dim=1)  # [b,4,h,w]
                # warp to a random novel view
                mask, warp_rgb, warp_d, real_img_pose, inpaint_img_pose = self.load_mask(rgbd)
                warp_gray = self.gray_tensor_transform(warp_rgb)  # [b,1,h,w]
                edge = self.load_edge(warp_gray, mask)  # [b, 1, h, w]
                # edge-connect inpaint
                warp_rgb, warp_d = self.edge_connect_inpaint(warp_rgb, warp_gray, warp_d, edge, 1 - mask)

                if self.gt_as_src:
                    occ_mask, rec_rgb, rec_d = self.get_occ_mask(rgbd.permute(0, 2, 3, 1), real_img_pose, inpaint_img_pose)
                    occ_mask = occ_mask.permute(0, 3, 1, 2)
                    
                    tgt_items, src_items = src_items, tgt_items

                    tgt_items["img"] = warp_rgb
                    tgt_items["depth"] = warp_d
                    
                    tgt_items["occ_mask"] = occ_mask
                    tgt_items["G_src_tgt"] = self.G_src_tgt.inverse().cuda(self.rank)
                else:
                    inpaint_rgbd = torch.cat([warp_rgb, warp_d], dim=1)
                    # warp back to obtain the occ_mask
                    # NOTE during training: the original img is the target view, the warped img is the source view
                    # occ_mask indicates the pixels in the original img (target view) 
                    # that are unseen in the source view (warp rgb)
                    occ_mask, rec_rgb, rec_d = self.get_occ_mask(inpaint_rgbd, inpaint_img_pose, real_img_pose)

                    src_items["img"] = warp_rgb
                    src_items["depth"] = warp_d
                    
                    tgt_items["occ_mask"] = occ_mask
                    tgt_items["G_src_tgt"] = self.G_src_tgt.cuda(self.rank)

                # # uncomment the following code to debug
                # bs = occ_mask.shape[0]
                # save_image(torch.cat([occ_mask.repeat(1, 3, 1, 1), 
                #                       rec_rgb, 
                #                       warp_rgb,
                #                       tgt_items["img"],
                #                       warp_d.repeat(1, 3, 1, 1),
                #                       tgt_items["depth"].repeat(1, 3, 1, 1)], dim=0), 'occ_mask.jpg', nrow=bs)
                # a = 10
                
        return names, src_items, tgt_items


if __name__ == '__main__':
    data_root = '/root/autodl-tmp/adampi-data/val2017'
    depth_root = '/root/autodl-tmp/adampi-data/depth/val2017'
    ec_weight_dir = "/root/autodl-tmp/adampi-data/ecweight"
    dataset = COCODataset(
        data_root, depth_root, 256, 384, debug=True, ec_weight_dir=ec_weight_dir, rank="cuda:1",
        trans_range={'x':0.2, 'y':0., 'z':0., 'a':-1, 'b':-1, 'c':-1},
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, shuffle=False)
    for item in loader:
        a = 10
