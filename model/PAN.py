import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(hidden_channels)
    
    def forward(self, x):
        return self.activation(self.conv3(x) + self.conv2(self.bn(self.activation(self.conv1(x)))))


class DownsizeEncoder(nn.Module):
    def __init__(self, num_blocks, dim_in, dim_out):
        super().__init__()
        res_blocks = []
        for i_block in range(0, num_blocks):
            d_in = dim_in if i_block == 0 else max(dim_in, dim_out // (2 ** (num_blocks - i_block)))
            d_out = max(dim_in, dim_out // (2 ** (num_blocks - i_block - 1)))
            res_blocks.append(ResBlock(in_channels=d_in, out_channels=d_out, hidden_channels=d_out))
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, x):
        # [b, c, h, w]
        for res_block in self.res_blocks:
            x = res_block(x)
            x = F.avg_pool2d(x, kernel_size=2)
        return x       # [b, c, h, w]


class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_qk, dim_v):
        super().__init__()
        self.wQs = nn.ModuleList([nn.Linear(dim_in, dim_qk) for _ in range(num_heads)])
        self.wKs = nn.ModuleList([nn.Linear(dim_in, dim_qk) for _ in range(num_heads)])
        self.wVs = nn.ModuleList([nn.Linear(dim_in, dim_v // num_heads) for _ in range(num_heads)])
        self.fusion = nn.Linear(dim_v, dim_v)
        self.norm = dim_qk ** 0.5

    def forward(self, feat):
        feat_atted = []
        for wQ, wK, wV in zip(self.wQs, self.wKs, self.wVs):
            Q = wQ(feat)  # [b,s,cq]
            K = wK(feat)  # [b,s,cq]
            V = wV(feat)  # [b,s,cv]
            att = torch.softmax(torch.einsum('bik,bjk->bij', Q, K) / self.norm, dim=2)
            feat_atted.append(torch.einsum('bij,bjc->bic', att, V))        
        return self.fusion(torch.cat(feat_atted, dim=-1))  # [b,s,c]


class LinearSigmoid(nn.Module):
    def __init__(self, in_ch, disp_range):
        super().__init__()
        self.start, self.end = disp_range
        self.linear = nn.Linear(in_ch, 1)

    def forward(self, feat, init_disp):
        feat = self.linear(feat).squeeze(-1)  # [b,s]
        return init_disp + feat * 1. / init_disp.shape[1] 


class DepthPredictionNetwork(nn.Module):
    def __init__(self, disp_range, **kwargs):
        super().__init__()
        self.context_encoder = DownsizeEncoder(num_blocks=5, dim_in=5, dim_out=128)
        self.self_attention = MultiheadSelfAttention(num_heads=4, dim_in=128, dim_qk=32, dim_v=128)
        self.embed = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.to_disp = LinearSigmoid(32, disp_range)

    def forward(self, init_disp, rgb_low_res, disp_low_res):
        B, S = init_disp.shape
        
        # context encoder
        x = torch.cat([
                rgb_low_res[:, None, ...].repeat(1, S, 1, 1, 1), 
                disp_low_res[:, None, ...].repeat(1, S, 1, 1, 1), 
                init_disp[:, :, None, None, None].repeat(1, 1, 1, *rgb_low_res.shape[-2:])
            ], dim=-3)   # [b, s, 5, h/4, w/4]
        x = x.view(-1, *x.shape[-3:])  # [b*s, 5, h/4, w/4]
        context = self.context_encoder(x)   # [b*s, c, h/128, w/128]
        context = F.adaptive_avg_pool2d(context, (1, 1)).squeeze(-1).squeeze(-1)  # [b*s, c]
        context = context.view(B, S, -1)  # [b, s, c]
        
        # self attention
        feat_atted = self.self_attention(context)   # [b, s, c ]
        feat = self.embed(feat_atted)  # [b, s, c]
        disp_bs = self.to_disp(feat, init_disp)  # [b, s]
        return disp_bs
