import argparse
import torch
import torch.nn.functional as F

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto,
)
from model.AdaMPI import MPIPredictor


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_path', type=str, default="images/0810.png")
parser.add_argument('--disp_path', type=str, default="images/depth/0810.png")
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--save_path', type=str, default="debug/0810.mp4")
parser.add_argument('--ckpt_path', type=str, default="adampiweight/adampi_64p.pth")
opt, _ = parser.parse_known_args()


# load input
image = image_to_tensor(opt.img_path).cuda()  # [1,3,h,w]
disp = disparity_to_tensor(opt.disp_path).cuda()  # [1,1,h,w]
image = F.interpolate(image, size=(opt.height, opt.width), mode='bilinear', align_corners=True)
disp = F.interpolate(disp, size=(opt.height, opt.width), mode='bilinear', align_corners=True)

# load pretrained model
ckpt = torch.load(opt.ckpt_path)
model = MPIPredictor(
    width=opt.width,
    height=opt.height,
    num_planes=ckpt["num_planes"],
)
model.load_state_dict(ckpt["weight"])
model = model.cuda()
model = model.eval()

# predict MPI planes
with torch.no_grad():
    pred_mpi_planes, pred_mpi_disp = model(image, disp)  # [b,s,4,h,w]

# render 3D photo
K = torch.tensor([
    [0.58, 0, 0.5],
    [0, 0.58, 0.5],
    [0, 0, 1]
]).cuda()
K[0, :] *= opt.width
K[1, :] *= opt.height
K = K.unsqueeze(0)

render_3dphoto(
    image,
    pred_mpi_planes,
    pred_mpi_disp,
    K,
    K,
    opt.save_path,
)
