import torch
import torchvision
import torch.nn.functional as F


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.resize = resize

    def forward(self, syn_imgs, gt_imgs):
        syn_imgs = (syn_imgs - self.mean.to(syn_imgs)) / self.std.to(syn_imgs)
        gt_imgs = (gt_imgs - self.mean.to(syn_imgs)) / self.std.to(syn_imgs)
        if self.resize:
            syn_imgs = self.transform(syn_imgs, mode="bilinear", size=(224, 224),
                                      align_corners=False)
            gt_imgs = self.transform(gt_imgs, mode="bilinear", size=(224, 224),
                                     align_corners=False)

        loss = 0.0
        x = syn_imgs
        y = gt_imgs
        for block in self.blocks:
            with torch.no_grad():
                x = block(x)
                y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def rank_loss_fn(disps: torch.Tensor, start=1, end=0.0001, min_gap: float = 0.):
    # disp: [B, S]
    return torch.mean(F.relu(disps[:, 1:] - disps[:, :-1] + min_gap)) + torch.mean(F.relu(end - disps)) + torch.mean(F.relu(disps - start))


def assign_loss(feat_mask, mpi_disp, depth):
    '''
    feat_mask: [b,s,h,w]
    mpi_disp: [b,s]
    depth: [b,1,h,w]
    '''
    dist = torch.abs(feat_mask * (depth - mpi_disp[..., None, None]))  # [b,s,h,w]
    return torch.mean(dist)
