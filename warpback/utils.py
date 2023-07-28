import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes


class RGBDRenderer:
    def __init__(self, device):
        self.device = device
        self.eps = 1e-4
        self.near_z = 1e-4
        self.far_z = 1e4
    
    def render_mesh(self, mesh_dict, cam_int, cam_ext):
        '''
        input:
            mesh: the output for construct_mesh function
            cam_int: [b,3,3] normalized camera intrinsic matrix
            cam_ext: [b,3,4] camera extrinsic matrix with the same scale as depth map
            camera coord: x to right, z to front, y to down
        
        output:
            render: [b,3,h,w]
            disparity: [b,1,h,w]
        '''
        vertice = mesh_dict["vertice"]  # [b,h*w,3]
        faces = mesh_dict["faces"]  # [b,nface,3]
        attributes = mesh_dict["attributes"]  # [b,h*w,4]
        h, w = mesh_dict["size"]

        ############
        # to NDC space
        vertice_homo = self.lift_to_homo(vertice)  # [b,h*w,4]
        # [b,1,3,4] x [b,h*w,4,1] = [b,h*w,3,1]
        vertice_world = torch.matmul(cam_ext.unsqueeze(1), vertice_homo[..., None]).squeeze(-1)  # [b,h*w,3]
        vertice_depth = vertice_world[..., -1:]  # [b,h*w,1]
        attributes = torch.cat([attributes, vertice_depth], dim=-1)  # [b,h*w,5]
        # [b,1,3,3] x [b,h*w,3,1] = [b,h*w,3,1]
        vertice_world_homo = self.lift_to_homo(vertice_world)
        persp = self.get_perspective_from_intrinsic(cam_int)  # [b,4,4]

        # [b,1,4,4] x [b,h*w,4,1] = [b,h*w,4,1]
        vertice_ndc = torch.matmul(persp.unsqueeze(1), vertice_world_homo[..., None]).squeeze(-1)  # [b,h*w,4]
        vertice_ndc = vertice_ndc[..., :-1] / vertice_ndc[..., -1:]
        vertice_ndc[..., :-1] *= -1
        vertice_ndc[..., 0] *= w / h

        ############
        # render
        mesh = Meshes(vertice_ndc, faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(mesh, (h, w), faces_per_pixel=1, blur_radius=1e-6)  # [b,h,w,1] [b,h,w,1,3]

        b, nf, _ = faces.size()
        faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, 5)  # [b,3f,5]
        face_attributes = torch.gather(attributes, dim=1, index=faces)  # [b,3f,5]
        face_attributes = face_attributes.reshape(b * nf, 3, 5)
        output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
        output = output.squeeze(-2).permute(0, 3, 1, 2)

        render = output[:, :3]
        mask = output[:, 3:4]
        disparity = torch.reciprocal(output[:, 4:] + self.eps)
        return render * mask, disparity * mask, mask

    def construct_mesh(self, rgbd, cam_int):
        '''
        input:
            rgbd: [b,4,h,w]
                the first 3 channels for RGB
                the last channel for normalized disparity, in range [0,1]
            cam_int: [b,3,3] normalized camera intrinsic matrix
        
        output:
            mesh_dict: define mesh in camera space, includes the following keys
                vertice: [b,h*w,3]
                faces:  [b,nface,3]
                attributes: [b,h*w,c] include color and mask
        '''
        b, _, h, w = rgbd.size()
        
        ############
        # get pixel coordinates
        pixel_2d = self.get_screen_pixel_coord(h, w)  # [1,h,w,2]
        pixel_2d_homo = self.lift_to_homo(pixel_2d)  # [1,h,w,3]

        ############
        # project pixels to 3D space
        rgbd = rgbd.permute(0, 2, 3, 1)  # [b,h,w,4]
        disparity = rgbd[..., -1:]  # [b,h,w,1]
        depth = torch.reciprocal(disparity + self.eps)  # [b,h,w,1]
        cam_int_inv = torch.inverse(cam_int)  # [b,3,3]
        # [b,1,1,3,3] x [1,h,w,3,1] = [b,h,w,3,1]
        pixel_3d = torch.matmul(cam_int_inv[:, None, None, :, :], pixel_2d_homo[..., None]).squeeze(-1)  # [b,h,w,3]
        pixel_3d = pixel_3d * depth  # [b,h,w,3]
        vertice = pixel_3d.reshape(b, h * w, 3)  # [b,h*w,3]

        ############
        # construct faces
        faces = self.get_faces(h, w)  # [1,nface,3]
        faces = faces.repeat(b, 1, 1).long()  # [b,nface,3]

        ############
        # compute attributes
        attr_color = rgbd[..., :-1].reshape(b, h * w, 3)  # [b,h*w,3]
        attr_mask = self.get_visible_mask(disparity).reshape(b, h * w, 1)  # [b,h*w,1]
        attr = torch.cat([attr_color, attr_mask], dim=-1)  # [b,h*w,4]

        mesh_dict = {
            "vertice": vertice,
            "faces": faces,
            "attributes": attr,
            "size": [h, w],
        }
        return mesh_dict

    def get_screen_pixel_coord(self, h, w):
        '''
        get normalized pixel coordinates on the screen
        x to left, y to down
        
        e.g.
        [0,0][1,0][2,0]
        [0,1][1,1][2,1]
        output:
            pixel_coord: [1,h,w,2]
        '''
        x = torch.arange(w).to(self.device)  # [w]
        y = torch.arange(h).to(self.device)  # [h]
        x = (x + 0.5) / w
        y = (y + 0.5) / h
        x = x[None, None, ..., None].repeat(1, h, 1, 1)  # [1,h,w,1]
        y = y[None, ..., None, None].repeat(1, 1, w, 1)  # [1,h,w,1]
        pixel_coord = torch.cat([x, y], dim=-1)  # [1,h,w,2]
        return pixel_coord
    
    def lift_to_homo(self, coord):
        '''
        return the homo version of coord
        input: coord [..., k]
        output: homo_coord [...,k+1]
        '''
        ones = torch.ones_like(coord[..., -1:])
        return torch.cat([coord, ones], dim=-1)

    def get_faces(self, h, w):
        '''
        get face connect information
        x to left, y to down
        e.g.
        [0,0][1,0][2,0]
        [0,1][1,1][2,1]
        faces: [1,nface,3]
        '''
        x = torch.arange(w - 1).to(self.device)  # [w-1]
        y = torch.arange(h - 1).to(self.device)  # [h-1]
        x = x[None, None, ..., None].repeat(1, h - 1, 1, 1)  # [1,h-1,w-1,1]
        y = y[None, ..., None, None].repeat(1, 1, w - 1, 1)  # [1,h-1,w-1,1]

        tl = y * w + x
        tr = y * w + x + 1
        bl = (y + 1) * w + x
        br = (y + 1) * w + x + 1

        faces_l = torch.cat([tl, bl, br], dim=-1).reshape(1, -1, 3)  # [1,(h-1)(w-1),3]
        faces_r = torch.cat([br, tr, tl], dim=-1).reshape(1, -1, 3)  # [1,(h-1)(w-1),3]

        return torch.cat([faces_l, faces_r], dim=1)  # [1,nface,3]

    def get_visible_mask(self, disparity, beta=10, alpha_threshold=0.3):
        '''
        filter the disparity map using sobel kernel, then mask out the edge (depth discontinuity)
        input:
            disparity: [b,h,w,1]
        
        output:
            vis_mask: [b,h,w,1]
        '''
        b, h, w, _ = disparity.size()
        disparity = disparity.reshape(b, 1, h, w)  # [b,1,h,w]
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        sobel_x = F.conv2d(disparity, kernel_x, padding=(1, 1))  # [b,1,h,w]
        sobel_y = F.conv2d(disparity, kernel_y, padding=(1, 1))  # [b,1,h,w]
        sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(b, h, w, 1)  # [b,h,w,1]
        alpha = torch.exp(-1.0 * beta * sobel_mag)  # [b,h,w,1]
        vis_mask = torch.greater(alpha, alpha_threshold).float()
        return vis_mask

    def get_perspective_from_intrinsic(self, cam_int):
        '''
        input:
            cam_int: [b,3,3]
        
        output:
            persp: [b,4,4]
        '''
        fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]  # [b]
        cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]  # [b]

        one = torch.ones_like(cx)  # [b]
        zero = torch.zeros_like(cx)  # [b]

        near_z, far_z = self.near_z * one, self.far_z * one
        a = (near_z + far_z) / (far_z - near_z)
        b = -2.0 * near_z * far_z / (far_z - near_z)

        matrix = [[2.0 * fx, zero, 2.0 * cx - 1.0, zero],
                  [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
                  [zero, zero, a, b],
                  [zero, zero, one, zero]]
        # -> [[b,4],[b,4],[b,4],[b,4]] -> [b,4,4]        
        persp = torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)  # [b,4,4]
        return persp


#######################
# some helper I/O functions
#######################
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


#######################
# some helper geometry functions
# adapt from https://github.com/mattpoggi/depthstillation
#######################
def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


#######################
# some helper dataset preprocess functions
#######################
def bleed(x, mask):
    B, C, H, W = x.shape
    kernel = torch.tensor([[0.5, 1., 0.], [1., 1., 1.], [0.5, 1., 0.5]]).view(1, 1, 3, 3).to(x)
    rmask = 1. - mask
    return rmask * x  + mask * F.conv2d((x * rmask).view(B * C, 1, H, W), kernel, padding=1).view(B, C, H, W) / (F.conv2d(rmask, kernel, padding=1) + 1e-5)


def shrink_mask(mask):
    kernel = torch.tensor([[0.5, 1., 0.], [1., 1., 1.], [0.5, 1., 0.5]]).view(1, 1, 3, 3).to(mask)
    return (F.conv2d(mask, kernel, padding=1) > kernel.sum() - 1e-3).float()


def seamless_clone(source, target, mask):
    B, C, H, W = source.shape
    source, target = source.contiguous(), target.contiguous()
    laplacian_kernel = torch.tensor([[0., -0.25, 0], [-0.25, 1., -0.25], [0., -0.25, 0.]]).view(1, 1, 3, 3).to(source)
    coef_kernel = torch.tensor([[0., 0.25, 0], [0.25, 0., 0.25], [0., 0.25, 0.]]).view(1, 1, 3, 3).to(source)

    mixed = None
    for scale in range(3, 0, -1):
        H_, W_ = H // scale, W // scale
        target_ = F.interpolate(target, size=(H_, W_))
        mask_ = F.interpolate(mask, size=(H_, W_))
        source_ = F.interpolate(source, size=(H_, W_))
        target_ = bleed(target_, mask_)
        mask_ = shrink_mask(mask_)

        source_laplacian = F.conv2d(source_.view(B * C, 1, H_, W_), laplacian_kernel, padding=1).view(B, C, H_, W_)
        if mixed is None:
            mixed =  target_ * (1. - mask_) + source_ * mask_
        else:
            mixed = target_ * (1. - mask_) + F.interpolate(mixed, size=(H_, W_)) * mask_
        for _ in range(4 if scale > 1 else 8):
            mixed = mixed * (1. - mask_) + (F.conv2d(mixed.view(B * C, 1, H_, W_), coef_kernel, padding=1).view(B, C, H_, W_) + source_laplacian) * mask_

    return mixed


if __name__ == "__main__":
    device = "cuda"
    render_save_path = "debug/render_rgb.png"
    disp_save_path = "debug/render_depth.png"
    img_paths = ["warpback/toydata/000000013774.jpg", "warpback/toydata/000000017959.jpg"]
    disp_paths = ["warpback/toydata/dpt_depth/000000013774.png", "warpback/toydata/dpt_depth/000000017959.png"]

    h, w = 256, 384

    cam_int = torch.tensor([[0.58, 0, 0.5],
                            [0, 0.58, 0.5],
                            [0, 0, 1]]).to(device)
    cam_ext = torch.tensor([[1., 0., 0., 0.2],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.]]).to(device)

    bs = len(img_paths)
    cam_int = cam_int[None, ...].repeat(bs, 1, 1)  # [b,3,3]
    cam_ext = cam_ext[None, ...].repeat(bs, 1, 1)  # [b,3,3]

    rgbd = []
    for ip, dp in zip(img_paths, disp_paths):
        cur = torch.cat([image_to_tensor(ip), disparity_to_tensor(dp)], dim=1)
        cur = F.interpolate(cur, size=(h, w), mode="bilinear", align_corners=False)
        rgbd.append(cur)
    rgbd = torch.cat(rgbd, dim=0).to(device)  # [b,4,h,w]

    rgbd_renderer = RGBDRenderer(device)
    mesh = rgbd_renderer.construct_mesh(rgbd, cam_int)
    render, disp, _ = rgbd_renderer.render_mesh(mesh, cam_int, cam_ext)
    save_image(render, render_save_path)
    save_image(disp, disp_save_path)
