import math
import warnings
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class UpMethod(nn.Module):
    def __init__(self, decoder, **kwargs):
        super().__init__()
        self.method = decoder
        if decoder is None:
            self.up_method = nn.Identity()
        elif decoder == 'interpolate':
            out_size = (2 * kwargs['T'], kwargs['V'])
            self.up_method = nn.Upsample(size=out_size, mode=kwargs['mode'])
        elif decoder == 'linear':
            in_size = kwargs['T'] * kwargs['V']
            self.up_method = nn.Linear(in_features=in_size, out_features=2 * in_size)
        elif decoder == 'conv':
            self.up_method = nn.Conv2d(kwargs['T'], 2 * kwargs['T'], 1)

    def forward(self, x):
        N, C, T, V = x.shape
        if self.method == 'linear':
            x = x.view(N, C, T * V)
            x = self.up_method(x)
            x = x.view(N, C, 2 * T, V)
        elif self.method == 'conv':
            x = x.transpose(1, 2)
            x = self.up_method(x)
            x = x.transpose(1, 2)
        else:
            x = self.up_method(x)
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TempModule(nn.Sequential):
    def __init__(self, in_channels, branch_channels, ks, stride, dilation, T, V, decoder=None):
        super().__init__(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                padding=0
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            UpMethod(decoder, T=T, V=V, mode='bilinear'),
            TemporalConv(
                branch_channels,
                branch_channels,
                kernel_size=ks,
                stride=stride,
                dilation=dilation)
        )


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=False,
                 residual_kernel_size=1,
                 T=13,
                 V=137,
                 decoder=None):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        if type(decoder) == list:
            assert len(decoder) == len(dilations)
            temp_decoder = decoder
        else:
            temp_decoder = [decoder] * len(dilations)

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            TempModule(in_channels, branch_channels, ks, stride, dilation, T, V, decoder)
            for ks, dilation, decoder in zip(kernel_size, dilations, temp_decoder)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            UpMethod(decoder, T=T, V=V, mode='bilinear'),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            UpMethod(decoder, T=T, V=V, mode='bilinear'),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class MHSA(nn.Module):
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 insert_cls_layer=0, pe=False, num_point=25,
                 outer=True, layer=0,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer

        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None for _ in range(num_point)]
        h[0] = np.eye(num_point)
        h[1] = h1
        self.hops = 0 * h[0]
        for i in range(2, num_point):
            h[i] = h[i - 1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(num_point - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i * h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()
        #
        self.rpe = nn.Parameter(torch.zeros((self.hops.max() + 1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))

        A = A.sum(0)
        A[:, :] = 0

        self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0),
                                  requires_grad=True)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)
        self.e_kv = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        self.e_q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim, dim, 1, groups=6)

        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        N, C, T, V = x.shape
        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]
        e_kv = self.e_kv(e).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        e_k, e_v = e_kv[0], kv[1]

        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
        e_q = self.e_q(e).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)

        pos_emb = self.rpe[self.hops]

        k_r = pos_emb.view(V, V, self.num_heads, self.dim // self.num_heads)
        b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)

        c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)

        d = torch.einsum("bthnc, bthmc->bthnm", e_q, e_k)

        a = q @ k.transpose(-2, -1)

        attn = a + b + c + d
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (self.alpha * attn + self.outer) @ v
        # x = (attn + self.outer) @ v

        e_attn = d * self.scale
        e_attn = e_attn.softmax(dim=-1)
        e_attn = self.attn_drop(e_attn)
        e = (self.beta * e_attn) @ e_v

        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        e = e.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x, e


# using conv2d implementation after dimension permutation
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 num_heads=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x = self.fc1(x.transpose(1,2)).transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x.transpose(1,2)).transpose(1,2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpLinear(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 num_heads=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features // 2)
        self.fc1_2 = nn.Linear(hidden_features // 2, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x [N, C, T, V]
        x = self.fc1(x)
        x = self.fc1_2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class unit_vit(nn.Module):
    def __init__(self, dim_in, dim, A, num_of_heads, add_skip_connection=True, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer=0, stride=1,
                 insert_cls_layer=0, pe=False, num_point=25, decoder=None, **kwargs):
        super().__init__()
        self.layer = layer
        self.stride = stride
        self.norm1 = norm_layer(dim_in)
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        self.attn = MHSA(dim_in, dim, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop,
                         proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer,
                         **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.dim_in != self.dim:
            self.skip_proj = nn.Conv2d(dim_in, dim, (1, 1), padding=(0, 0), bias=False)
        self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False)
        self.pe = pe
        self.decoder = decoder

    def forward(self, x, joint_label, groups, e=None):
        if e is None or self.layer == 1 or e.shape[2] != x.shape[2]:
            if not joint_label:
                joint_label = [i for i in range(self.num_point)]
            # more efficient implementation
            label = F.one_hot(torch.tensor(joint_label)).float().to(x.device)
            z = x @ (label / label.sum(dim=0, keepdim=True))

            ########################################################################
            # w/o proj
            # z = z.permute(3, 0, 1, 2)
            # w/ proj
            z = self.pe_proj(z).permute(3, 0, 1, 2)

            # TODO 1 Eaug
            e = z[joint_label].permute(1, 2, 3, 0)
        # if self.decoder is None:
        #     # mask Ef
        #     e[..., groups[3]] = 0
        x_new, e_new = self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e)
        if self.add_skip_connection:
            if self.dim_in != self.dim:
                x = self.skip_proj(x) + self.drop_path(x_new)
            else:
                x = x + self.drop_path(x_new)
        else:
            x = self.drop_path(x_new)

        # x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))

        return x, e_new


class TCN_ViT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5,
                 dilations=[1, 2], pe=False, T=13, num_point=25, decoder=None, layer=0):
        super(TCN_ViT_unit, self).__init__()
        self.vit1 = unit_vit(in_channels, out_channels, A, add_skip_connection=residual, num_of_heads=num_of_heads,
                             pe=pe, num_point=num_point, decoder=decoder, layer=layer, stride=stride)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            # redisual=True has worse performance in the end
                                            residual=False, T=T, V=num_point, decoder=decoder)
        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            if decoder is None:
                self.residual = lambda x: x
            else:
                self.residual = UpMethod(decoder, T=T, V=num_point, mode='bilinear')

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups, e=None):
        x_new, e_new = self.vit1(x, joint_label, groups, e)
        y = self.act(self.tcn1(x_new) + self.residual(x))
        return y, e_new


# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size)
#         patch_size = (patch_size, patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # [B, 13*137, 216]
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # [B, 13*137, 216]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        B, C, T, V = x.shape
        x_norm1 = self.norm1(x.view(B, C, T * V).transpose(1, 2))
        x = x + self.drop_path(self.attn(x_norm1).transpose(1, 2).view(B, C, T, V))
        x_norm2 = self.norm2(x.view(B, C, T * V).transpose(1, 2))
        x = x + self.drop_path(self.mlp(x_norm2.transpose(1, 2).view(B, C, T, V)))
        return x


class Encoder(nn.Module):
    def __init__(self, num_of_heads, A, num_point):
        super().__init__()
        self.l1 = TCN_ViT_unit(3, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=1)
        # * num_heads, effect of concatenation following the official implementation
        self.l2 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=2)
        self.l3 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, stride=2,
                               num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=3)
        self.l4 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=4)
        self.l5 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, stride=2,
                               num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5)
        self.l6 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=6)

    def forward(self, x, joint_label, groups):
        x, e = self.l1(x, joint_label, groups)
        x, e = self.l2(x, joint_label, groups, e)
        x, e = self.l3(x, joint_label, groups, e)
        x, e = self.l4(x, joint_label, groups, e)
        x, e = self.l5(x, joint_label, groups, e)
        x, e = self.l6(x, joint_label, groups, e)
        return x


class Decoder(nn.Module):
    def __init__(self, num_of_heads, A, T, num_point, decoder):
        super().__init__()
        self.l1 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, T=T, num_point=num_point, decoder=decoder, layer=1)
        self.l2 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, T=T * 2, num_point=num_point, decoder=decoder, layer=2)
        self.rec_pe = nn.Conv2d(24 * num_of_heads, 3, kernel_size=(1, 1), bias=False)

    def forward(self, x, joint_label, groups):
        # [N, 216, 13, 137] -> [N, 216, 26, 137]
        x, e = self.l1(x, joint_label, groups)
        # [N, 216, 26, 137] -> [N, 216, 52, 137]
        x, e = self.l2(x, joint_label, groups, e)
        # [N, 216, 52, 137] -> [N, 216, 3, 137]
        x = self.rec_pe(x)
        return x


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=20, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, num_of_heads=9, joint_label=[], T=52, decoder=None, arg=None):
        super(Model, self).__init__()

        self.num_of_heads = num_of_heads
        self.num_class = num_class
        self.num_point = num_point
        self.num_epoch = arg.num_epoch
        self.seed = arg.seed
        self.mask_mode = arg.mask_mode
        if self.mask_mode is not None:
            self.mask_id = arg.mask_id
            self.mask_ratio = arg.mask_ratio
            indexes = [i for i in range(self.num_point)]
            num_mask = int(self.num_point * self.mask_ratio)
            random.seed(self.seed)
            self.index_mask = random.sample(indexes, num_mask)
        else:
            self.mask_ratio = 0

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args, index_mask=None)
            # self.graph_mask = Graph(**graph_args, index_mask=self.index_mask)

        A = self.graph.A  # 3,num_point,num_point
        # A_mask = self.graph_mask.A  # 3,num_point,num_point


        # self.num_point = num_point//2
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * self.num_point)
        # self.joint_label = joint_label[:num_point//2]
        self.joint_label = joint_label

        self.encoder = Encoder(num_of_heads, A, num_point)
        self.block = Block(24 * num_of_heads, num_of_heads)
        # standard ce loss
        self.fc = nn.Linear(24 * num_of_heads, num_class)

        if decoder is not None:
            # TODO frame downsample number
            self.decoder = Decoder(num_of_heads, A, T // 4, num_point, decoder)
        else:
            self.decoder = None

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # self.select_points = SelectPoint(num_point,num_point/2)
        # self.W = PointS(num_point)

    def get_mask_ratio(self):
        return format(self.mask_ratio, '.4f')

    def masked_input(self, x, groups, epoch=None):
        if self.mask_mode == 'group':
            for i in self.mask_id:
                x[..., groups[i]] = 0
        elif self.mask_mode == 'random':
            if epoch is not None:
                indexes = [i for i in range(self.num_point)]
                start_ratio = 0.20
                end_ratio = 0.80
                alpha = epoch / (self.num_epoch - 1)
                self.mask_ratio = start_ratio + (end_ratio - start_ratio) * alpha
                num_mask = int(self.num_point * self.mask_ratio)
                random.seed(self.seed)
                index_mask = random.sample(indexes, num_mask)
            else:
                index_mask = self.index_mask
            x[..., index_mask] = 0
            # x[..., index_mask] = x[..., 0:1, index_mask]
        return x

    def forward(self, x, y, epoch=None):
        # x-> 16,3,52,137,1
        # x = self.W(x)

        # x-> 16,3,52,137,1
        groups = []
        # for num in range(max(self.joint_label) + 1):
        #     groups.append([ind for ind, element in enumerate(self.joint_label) if element == num])

        # groups[0] = groups[:x.shape[3]]

        # N -> batch_size
        # C -> channel
        # T -> frame
        # V -> joint
        # M -> 1
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        # n, c, t, v
        # x -> [16, 3, 52, 137]
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)
        x_ori = x.clone()

        # TODO mask the input
        # if self.training and self.mask_mode is not None:
        if self.mask_mode is not None:
            x = self.masked_input(x, groups, epoch)

        x_ec = self.encoder(x, self.joint_label, groups)  # [B, 216, 13, 137]

        # TODO headattn
        x = self.block(x_ec)
        # x = x_ec

        # N*M, C, T, V
        _, C, T, V = x.size()
        # spatial temporal average pooling
        x = x.view(N, M, C, -1)  # N, M, C, T*V
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x_cls = self.fc(x)  # C -> cls
        # if self.training and self.decoder is not None:
        if self.decoder is not None:
            x = self.decoder(x_ec, self.joint_label, groups)
            return x_cls, x, x_ori
        return x_cls


class PointS(nn.Module):
    def __init__(self, num_point):
        super(PointS, self).__init__()
        self.W = nn.Linear(num_point, int(num_point / 2), bias=True)

    def forward(self, x):
        x = x.transpose(3, 4)
        after_point = self.W(x)
        h_k = F.normalize(after_point, dim=-1)
        h_k = F.relu(h_k)
        h_k = h_k.permute(0, 1, 2, 4, 3)
        return h_k


class SelectPoint(nn.Module):
    def __init__(self, infeat, outfeat, device='cuda', use_bn=True, mean=False, add_self=False):
        super(SelectPoint, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.device = device
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0).to(self.device))

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(h_k.size(1).to(self.device))
            h_k = self.bn(h_k)
        if mask is not None:
            h_k = h_k * mask.unsqueeze(2).expand_as(h_k)

        return h_k


if __name__ == '__main__':
    input = torch.rand((1, 3, 52, 137, 1)).cuda().float()
    input = input.transpose(3, 4)
    model = nn.Linear(137, 137 // 2, bias=True).cuda()
    out = model(input)
    print(out)
