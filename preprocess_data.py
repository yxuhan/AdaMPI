from transformers import DPTForDepthEstimation, DPTImageProcessor
import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_root', type=str)
parser.add_argument('--save_root', type=str)
opt, _ = parser.parse_known_args()


save_root = opt.save_root
img_root = opt.img_root

os.makedirs(save_root, exist_ok=True)
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").cuda()
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
with torch.no_grad():
    for pth in tqdm(os.listdir(img_root)):
        try:
            img_path = os.path.join(img_root, pth)
            img_name = os.path.splitext(pth)[0]
            save_path = os.path.join(save_root, "%s.png" % img_name)
            img = Image.open(img_path)
            inputs = image_processor(images=img, return_tensors="pt")
            midas_depth = model(pixel_values=inputs['pixel_values'].cuda()).predicted_depth.unsqueeze(1)
            width, height = img.size
            # Dump depth map for debugging
            midas_depth_np = F.interpolate(midas_depth, size=(height, width), mode="bilinear", align_corners=True).squeeze().cpu().numpy()
            formatted = (midas_depth_np * 255 / np.max(midas_depth_np)).astype("uint8")
            Image.fromarray(formatted).save(save_path)
        except:
            pass
