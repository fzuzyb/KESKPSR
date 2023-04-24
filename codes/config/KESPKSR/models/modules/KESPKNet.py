import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from utils import get_uperleft_denominator
from .module_util import *
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange





class TPAB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4,):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(nf2//2, nf2//2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf2//2, nf2//2, ksize2, 1, ksize2 // 2),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(nf2 // 2, nf2 // 2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf2 // 2, nf2 // 2, ksize2, 1, ksize2 // 2),
        )
        self.ESA_body3 = nn.Sequential(
            nn.Conv2d(nf2//2, nf2//2, 1),
            ESA(16, nf2//2)
        )
        self.CA_body1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1+nf2, nf1, ksize1, 1, ksize1 // 2),
            CALayer(nf1, 4))

        self.CA_body2 = CALayer(nf2//2, 4)

    def forward(self, x):

        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        f3 = self.body3(x[2])

        ca_f1 = self.CA_body1(torch.cat([f1, f2, f3], dim=1))
        ca_f2 = self.CA_body2(f2)
        ca_f3 = self.ESA_body3(f3)
        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2

        x[2] = x[2] + ca_f3

        return x
class GTFB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4,):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(nf2//2, nf2//2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf2//2, nf2//2, ksize2, 1, ksize2 // 2),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(nf2 // 2, nf2 // 2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf2 // 2, nf2 // 2, ksize2, 1, ksize2 // 2),
        )
        self.ESA_body3 = nn.Sequential(

            nn.Conv2d(nf2//2, nf2//2, 1),

            ESA(16, nf2//2)
        )
        self.CA_body1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1+nf2, nf1, ksize1, 1, ksize1 // 2),
            CALayer(nf1, 4))
        self.gla_body1 =GLA(n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=128, res_scale=1)
        self.CA_body2 = CALayer(nf2//2, 4)

    def forward(self, x):

        f1 = self.body1(x[0])
        f2 = self.body2(x[1])
        f3 = self.body3(x[2])

        ca_f1 = self.CA_body1(torch.cat([f1, f2, f3], dim=1))
        ca_f1 = self.gla_body1(ca_f1)

        ca_f2 = self.CA_body2(f2)
        ca_f3 = self.ESA_body3(f3)
        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2
        x[2] = x[2] + ca_f3

        return x


class TPAG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb,n_B):
        super().__init__()

        self.body = nn.Sequential(*[TPAB(nf1, nf2, ksize1, ksize2, ) for _ in range(nb)],
                                  *[GTFB(nf1, nf2, ksize1, ksize2, ) for _ in range(n_B)],
                                  )

    def forward(self, x):
        y = self.body(x)

        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        y[2] = x[2] + y[2]
        return y


class CLS(nn.Module):
    def __init__(self, nf, reduction=4):
        super().__init__()

        self.reduce_feature = nn.Conv2d(nf, nf//reduction, 1, 1, 0)

        self.grad_filter = nn.Sequential(
            nn.Conv2d(nf//reduction, nf//reduction, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf//reduction, nf//reduction, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf//reduction, nf//reduction, 3),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(nf//reduction, nf//reduction, 1),
        )

        self.expand_feature = nn.Conv2d(nf//reduction, nf, 1, 1, 0)

    def forward(self, x, kernel):
        cls_feats = self.reduce_feature(x)
        kernel_P = torch.exp(self.grad_filter(cls_feats))
        kernel_P = kernel_P - kernel_P.mean(dim=(2, 3), keepdim=True)
        clear_features = torch.zeros(cls_feats.size()).to(x.device)
        ks = kernel.shape[-1]
        dim = (ks, ks, ks, ks)
        feature_pad = F.pad(cls_feats, dim, "replicate")
        for i in range(feature_pad.shape[1]):
            feature_ch = feature_pad[:, i:i+1, :, :]
            clear_feature_ch = get_uperleft_denominator(feature_ch, kernel, kernel_P[:, i:i+1, :, :])
            clear_features[:, i:i+1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]

        x = self.expand_feature(clear_features)

        return x


class Estimator(nn.Module):
    def __init__(
        self, in_nc=1, nf=64, para_len=10, num_blocks=3, kernel_size=4, filter_structures=[]
    ):
        super(Estimator, self).__init__()

        self.filter_structures = filter_structures
        self.ksize = kernel_size
        self.G_chan = 16
        self.in_nc = in_nc
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3)
        )

        self.body = nn.Sequential(
            make_layer(basic_block, num_blocks)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, para_len, 1),
            nn.Flatten(),
        )

        self.dec = nn.ModuleList()
        for i, f_size in enumerate(self.filter_structures):
            if i == 0:
                in_chan = in_nc
            elif i == len(self.filter_structures) - 1:
                in_chan = in_nc
            else:
                in_chan = self.G_chan
            self.dec.append(nn.Linear(para_len, self.G_chan * in_chan * f_size**2))

        self.apply(initialize_weights)

    def calc_curr_k(self, kernels, batch):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.ones([1, batch*self.in_nc]).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(kernels):
            curr_k = F.conv2d(delta, w, padding=self.ksize - 1, groups=batch) if ind == 0 else F.conv2d(curr_k, w, groups=batch)
        curr_k = curr_k.reshape(batch, self.in_nc, self.ksize, self.ksize).flip([2, 3])
        return curr_k

    def forward(self, LR):
        batch, channel = LR.shape[0:2]
        f1 = self.head(LR)
        f = self.body(f1) + f1

        latent_kernel = self.tail(f)

        kernels = [self.dec[0](latent_kernel).reshape(
                                                batch*self.G_chan,
                                                channel,
                                                self.filter_structures[0],
                                                self.filter_structures[0])]

        for i in range(1, len(self.filter_structures)-1):
            kernels.append(self.dec[i](latent_kernel).reshape(
                                                batch*self.G_chan,
                                                self.G_chan,
                                                self.filter_structures[i],
                                                self.filter_structures[i]))

        kernels.append(self.dec[-1](latent_kernel).reshape(
                                                batch*channel,
                                                self.G_chan,
                                                self.filter_structures[-1],
                                                self.filter_structures[-1]))
        K = self.calc_curr_k(kernels, batch).mean(dim=1, keepdim=True)

        # for anisox2
        # K = F.softmax(K.flatten(start_dim=1), dim=1)
        # K = K.view(batch, 1, self.ksize, self.ksize)

        K = K / torch.sum(K, dim=(2, 3), keepdim=True)

        return K






class Restorer(nn.Module):
    def __init__(
        self, in_nc=1, nf=64, nb=8,n_B=1, ng=1, scale=4, input_para=10, reduction=4, min=0.0, max=1.0,


    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        out_nc = in_nc
        nf2 = nf // reduction
        self.conv_first = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)

        self.feature_block = make_layer(basic_block, 3)
        self.head1 = nn.Conv2d(nf, nf2, 3, 1, 1)
        self.head2 = CLS(nf, reduction=4)
        body = [TPAG(nf, nf2, 3, 3, nb,n_B) for _ in range(ng)]

        self.body = nn.Sequential(*body)
        self.fusion = nn.Conv2d(nf+nf2, nf, 3, 1, 1)
        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(nf,nf * scale,3,1,1,bias=True),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf,nf * scale,3,1,1,bias=True),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(nf,nf * scale ** 2,3,1,1,bias=True),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )

    def forward(self, input, kernel):

        f = self.conv_first(input)

        feature = self.feature_block(f)

        f1 = self.head1(feature)
        f2 = self.head2(feature, kernel)

        gy_DPAB,gy_trans=f1.chunk(2, dim=1)
        inputs = [f2, gy_DPAB,gy_trans]
        f2, f1,f0 = self.body(inputs)
        f = self.fusion(torch.cat([f1, f2, f0], dim=1)) + f
        out = self.upscale(f)

        return torch.clamp(out, min=self.min, max=self.max)


class KESPKNet(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=16,
        n_B=1,
        ng=5,
        in_nc=3,
        reduction=4,
        upscale=4,
        input_para=128,
        kernel_size=21,pca_matrix_path=None,

    ):
        super(KESPKNet, self).__init__()

        self.ksize = kernel_size
        self.scale = upscale

        if kernel_size == 21:
            filter_structures = [11, 7, 5, 1] # for iso kernels all
        elif kernel_size == 11:
            filter_structures = [7, 3, 3, 1] # for aniso kernels x2
        elif kernel_size == 31:
            filter_structures = [11, 9, 7, 5, 3] # for aniso kernels x4
        else:
            print("Please check your kernel size, or reset a group filters for DDLK")

        self.Restorer = Restorer(
        nf=nf, in_nc=in_nc, nb=nb,n_B=n_B, ng=ng, scale=self.scale, input_para=input_para, reduction=reduction
        )
        self.Estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, nf=nf, filter_structures=filter_structures

        )

    def forward(self, lr):

        kernel = self.Estimator(lr)

        sr = self.Restorer(lr, kernel.detach())

        return sr, kernel
