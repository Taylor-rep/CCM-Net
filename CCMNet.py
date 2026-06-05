import math
import typing as t
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from mamba_ssm import Mamba
# --- Efficient Feature Calibration Components ---

def get_skin_scans(H, W, device):
    """
    Generates indices for Lesion-Centric and Spiral scans.
    Returns:
        spiral_idx: Center-out spiral indices
        radial_idx: Center-out radial indices (distance based)
    """
    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_y, center_x = H // 2, W // 2

    # 1. Radial Scan (Distance-based)
    dist = (y - center_y)**2 + (x - center_x)**2
    radial_idx = dist.flatten().argsort()

    # 2. Spiral Scan (Center-out)
    # A simple way to get spiral: sort by (max(abs(dx), abs(dy)), angle)
    dy = y - center_y
    dx = x - center_x
    # Manhattan distance rings + angle for continuity
    ring = torch.max(torch.abs(dy), torch.abs(dx))
    angle = torch.atan2(dy.float(), dx.float())
    # Sort by ring first, then angle to create a spiral effect
    combined = ring.float() * 10.0 + angle
    spiral_idx = combined.flatten().argsort()

    return spiral_idx, radial_idx

class SkinMambaLayer(nn.Module):
    """
    Advanced Mamba Layer for Skin Lesion (ISIC) with:
    1. Horizontal Scan
    2. Vertical Scan
    3. Center-out Spiral Scan (Spatial Continuity)
    4. Center-out Radial Scan (Lesion-Centric)
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert input_dim % 4 == 0, "input_dim must be divisible by 4"
        self.mamba_dim = input_dim // 4

        # 4 Mamba branches for 4 different scan patterns
        self.mamba_h = Mamba(d_model=self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v = Mamba(d_model=self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_spiral = Mamba(d_model=self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_radial = Mamba(d_model=self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim),
        )

        self.proj = nn.Conv2d(input_dim, output_dim, 1, bias=False)
        self.norm = nn.LayerNorm(input_dim)
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Cache for indices
        self.register_buffer("indices_cache", None, persistent=False)
        self.cache_res = (0, 0)

    def _get_indices(self, H, W, device):
        if self.cache_res != (H, W):
            spiral_idx, radial_idx = get_skin_scans(H, W, device)
            # Store as buffer to move with model
            indices = torch.stack([spiral_idx, radial_idx], dim=0)
            self.indices_cache = indices
            self.cache_res = (H, W)
        return self.indices_cache[0], self.indices_cache[1]

    def forward(self, x):
        B, C, H, W = x.shape
        x_res = x

        # Flatten and normalize
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.norm(x_flat)

        # Split channels
        chunks = torch.chunk(x_flat, 4, dim=2)
        x_h, x_v, x_spiral, x_radial = chunks

        # 1. Horizontal Scan (Identity)
        out_h = self.mamba_h(x_h)

        # 2. Vertical Scan (Transpose)
        x_v_2d = x_v.reshape(B, H, W, -1).permute(0, 2, 1, 3)
        out_v = self.mamba_v(x_v_2d.reshape(B, W * H, -1))
        out_v = out_v.reshape(B, W, H, -1).permute(0, 2, 1, 3).reshape(B, H * W, -1)

        # 3. Spiral Scan
        spiral_idx, radial_idx = self._get_indices(H, W, x.device)

        # Permute to spiral sequence
        x_spiral_seq = x_spiral[:, spiral_idx, :]
        out_spiral_seq = self.mamba_spiral(x_spiral_seq)
        # Invert permutation
        inv_spiral_idx = torch.zeros_like(spiral_idx)
        inv_spiral_idx[spiral_idx] = torch.arange(H * W, device=x.device)
        out_spiral = out_spiral_seq[:, inv_spiral_idx, :]

        # 4. Radial Scan
        x_radial_seq = x_radial[:, radial_idx, :]
        out_radial_seq = self.mamba_radial(x_radial_seq)
        # Invert permutation
        inv_radial_idx = torch.zeros_like(radial_idx)
        inv_radial_idx[radial_idx] = torch.arange(H * W, device=x.device)
        out_radial = out_radial_seq[:, inv_radial_idx, :]

        # Fusion
        fused = torch.cat([out_h, out_v, out_spiral, out_radial], dim=2)
        fused = self.fusion_mlp(fused)

        # Output
        out = fused.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = self.proj(out)
        return out + self.skip_scale * x_res##x


class LSCM(nn.Module):
    """
    Local-Structural Contrast Module (LSCM)
    Focuses on lesion boundaries by calculating local-contrast features.
    Replaces PCGM/ScConv.
    """
    def __init__(self, dim):
        super().__init__()
        # Multi-scale local structural extraction
        self.dw_3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw_5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)

        # Adaptive Contrast Gate
        self.contrast_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # 1. Capture local structure at different scales
        s1 = self.dw_3x3(x)
        s2 = self.dw_5x5(x)

        # 2. Extract local contrast (Difference of Scales)
        contrast = torch.abs(s1 - s2)

        # 3. Channel-wise modulation
        gate = self.contrast_gate(contrast)
        out = (s1 + s2) * gate

        return self.proj(out) + x

class MGCM(nn.Module):
    """
    Mamba-Guided Contextual Modulation (MGCM)
    Uses global semantic context from Mamba to modulate local convolutional branches.
    """
    def __init__(self, mamba_dim, total_dim, num_local_branches=4):
        super().__init__()
        self.num_local_branches = num_local_branches
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Modulation generator: generates (scale, shift) for each branch
        # Input dim must match mamba_dim (gc), not total_in_channels
        mid_dim = max(4, mamba_dim // 2)
        self.mod_gen = nn.Sequential(
            nn.Linear(mamba_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_local_branches * 2),
        )

        self.final_proj = nn.Conv2d(total_dim, total_dim, 1)

    def forward(self, local_branches, mamba_feat):
        B, C_m, H, W = mamba_feat.shape

        # 1. Extract Global Semantic Context from Mamba
        global_context = self.gap(mamba_feat).view(B, -1) # [B, C_m]

        # 2. Generate Modulation Parameters
        params = self.mod_gen(global_context) # [B, num_local_branches * 2]
        params = params.view(B, self.num_local_branches, 2, 1, 1, 1)

        # 3. Apply Global-to-Local Modulation
        modulated_local = []
        for i in range(self.num_local_branches):
            scale = params[:, i, 0] + 1.0
            shift = params[:, i, 1]
            modulated_local.append(local_branches[i] * scale + shift)

        # 4. Synergistic Fusion
        combined = torch.cat(modulated_local + [mamba_feat], dim=1)
        return self.final_proj(combined)

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution with Mamba branch integrated with ScCalibration
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        # Calculate gc to be a multiple of 4 (required by DCMambaLayer)
        gc = (int(in_channels * branch_ratio) // 4) * 4
        gc = max(4, gc) # At least 4 channels for the 4 scanning directions

        # Ensure we don't exceed in_channels
        while gc * 4 > in_channels and gc > 4:
            gc -= 4

        # Final safety check
        if gc * 4 > in_channels:
            # Fallback if in_channels is too small for DCMambaLayer
            # In this extreme case, we might need a simpler Mamba implementation,
            # but for medical imaging UNet, channels are usually 32, 64, 128...
            gc = in_channels // 4
            gc = (gc // 4) * 4 # Must still be multiple of 4
            gc = max(0, gc)

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        # SkinMamba Branch: Optimized for ISIC (Center-out Spiral & Radial scans)
        self.skin_mamba = SkinMambaLayer(
            input_dim=gc,
            output_dim=gc,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # LSCM Branch: Local-Structural Contrast Module
        id_channels = in_channels - 4 * gc
        if id_channels > 0:
            self.lscm = LSCM(id_channels)
        else:
            self.lscm = nn.Identity()

        # MGCM: Mamba-Guided Contextual Modulation
        # Modulates 4 local branches using global mamba context (gc)
        self.mgcm = MGCM(mamba_dim=gc, total_dim=in_channels, num_local_branches=4)

        self.split_indexes = (id_channels, gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h, x_mamba = torch.split(x, self.split_indexes, dim=1)

        # 1. Local Structural Enhancement
        x_lscm = self.lscm(x_id)

        # 2. Local Convolutional Branches
        x_hw_res = self.dwconv_hw(x_hw)
        x_w_res = self.dwconv_w(x_w)
        x_h_res = self.dwconv_h(x_h)

        # 3. Global Mamba Branch
        x_mamba_res = self.skin_mamba(x_mamba)

        # 4. Mamba-Guided Contextual Modulation & Fusion
        # The Mamba feature guides how the local features are combined
        local_branches = [x_lscm, x_hw_res, x_w_res, x_h_res]
        return self.mgcm(local_branches, x_mamba_res)

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, guide_dim, num_heads=4, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(guide_dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(guide_dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        # x is the encoder feature, fg and bg are the decoupled features from deepest layer
        x = self.input_cbr(x)

        # Guidance features fg/bg need to match spatial size of x
        if fg.size(2) != x.size(2) or fg.size(3) != x.size(3):
            fg = F.interpolate(fg, size=x.shape[2:], mode='bilinear', align_corners=True)
            bg = F.interpolate(bg, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)

        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                              self.kernel_size * self.kernel_size,
                                                                              -1).permute(0, 1, 4, 3, 2)

        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)
        out = self.output_cbr(x_weighted_bg)
        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


class DecoupleLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoupleLayer, self).__init__()
        mid_c = in_c // 2 if in_c > 32 else in_c
        self.cbr_fg = nn.Sequential(
            CBR(in_c, mid_c, kernel_size=3, padding=1),
            CBR(mid_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, mid_c, kernel_size=3, padding=1),
            CBR(mid_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        # self.cbr_uc = nn.Sequential(
        #     CBR(in_c, mid_c, kernel_size=3, padding=1),
        #     CBR(mid_c, out_c, kernel_size=3, padding=1),
        #     CBR(out_c, out_c, kernel_size=1, padding=0)
        # )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        # f_uc = self.cbr_uc(x)
        return f_fg, f_bg

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.bn(x))

class Down2(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)

    def forward(self, x):
        return self.conv(self.bn(x))

class ConvLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0, stride=1)
        self.act1 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0, stride=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
# class Boundary_Prediction_Generator(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

#     def forward(self, x):
#         boundary = torch.sigmoid(self.conv(x))
#         x = x + x * boundary
#         return x, boundary

class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, gt_pre

class Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        boundary = torch.sigmoid(self.conv1(x))#边界
        gt_pre = self.conv2(x)
        return (x + x * boundary + x * torch.sigmoid(gt_pre)), gt_pre, boundary

class Group_shuffle_block(nn.Module):#编码器和解码器模块
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.conv = nn.Sequential(
            InceptionDWConv2d(dim_in),
            nn.BatchNorm2d(dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class GuidedFusion(nn.Module):
    """
    Advanced fusion module with spatial and channel attention, guided by a prediction map.
    """
    def __init__(self, dim_decoder, dim_encoder, dim_out):
        super().__init__()
        # Convolutions to project encoder and decoder features to a common dimension
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(dim_encoder, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

        # Spatial gating mechanism learns attention from encoder features and a guidance map
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim_out + 1, dim_out // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out // 2, dim_out, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel attention mechanism for the fused features
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out, dim_out // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out // 4, dim_out, kernel_size=1),
            nn.Sigmoid()
        )

        # Final convolution to refine the fused and attended features
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_decoder, x_encoder, guidance_map):
        x_enc_proj = self.conv_encoder(x_encoder)
        x_dec_proj = self.conv_decoder(x_decoder)

        # Apply spatial gate using the guidance map
        spatial_gate_input = torch.cat([x_enc_proj, guidance_map], dim=1)
        spatial_attn = self.spatial_gate(spatial_gate_input)
        gated_encoder_feat = x_enc_proj * spatial_attn

        # Fuse decoder and gated encoder features
        fused_feat = x_dec_proj + gated_encoder_feat

        # Apply channel attention
        channel_attn = self.channel_gate(fused_feat)
        refined_feat = fused_feat * channel_attn

        output = self.final_conv(refined_feat)
        return output

class GuidedFusion2(nn.Module):
    """
    GuidedFusion variant that accepts two guidance maps (e.g., segmentation and boundary).
    """
    def __init__(self, dim_decoder, dim_encoder, dim_out):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(dim_encoder, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

        # Spatial gate now takes two guidance maps
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim_out + 2, dim_out // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out // 2, dim_out, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out, dim_out // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out // 4, dim_out, kernel_size=1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_decoder, x_encoder, guidance_map1, guidance_map2):
        x_enc_proj = self.conv_encoder(x_encoder)
        x_dec_proj = self.conv_decoder(x_decoder)

        # Apply spatial gate using two guidance maps
        spatial_gate_input = torch.cat([x_enc_proj, guidance_map1, guidance_map2], dim=1)
        spatial_attn = self.spatial_gate(spatial_gate_input)
        gated_encoder_feat = x_enc_proj * spatial_attn

        fused_feat = x_dec_proj + gated_encoder_feat

        channel_attn = self.channel_gate(fused_feat)
        refined_feat = fused_feat * channel_attn

        output = self.final_conv(refined_feat)
        return output

class CCMNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64]):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            ConvLayer(c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            Group_shuffle_block(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Group_shuffle_block(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Group_shuffle_block(c_list[4], c_list[5]),
        )


        self.Down1 = Down(c_list[0])#图像大小减半
        self.Down2 = Down(c_list[1])
        self.Down3 = Down(c_list[2])

        self.merge1 = GuidedFusion2(dim_decoder=c_list[0], dim_encoder=c_list[0], dim_out=c_list[0])
        self.merge2 = GuidedFusion2(dim_decoder=c_list[1], dim_encoder=c_list[1], dim_out=c_list[1])
        self.merge3 = GuidedFusion2(dim_decoder=c_list[2], dim_encoder=c_list[2], dim_out=c_list[2])
        self.merge4 = GuidedFusion(dim_decoder=c_list[3], dim_encoder=c_list[3], dim_out=c_list[3])
        self.merge5 = GuidedFusion(dim_decoder=c_list[4], dim_encoder=c_list[4], dim_out=c_list[4])

        self.decoder1 = nn.Sequential(
            Group_shuffle_block(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            Group_shuffle_block(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            Group_shuffle_block(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.pred1 = Image_Prediction_Generator(c_list[4])
        self.pred2 = Image_Prediction_Generator(c_list[3])
        self.gate1 = Prediction_Generator(c_list[2])
        self.gate2 = Prediction_Generator(c_list[1])
        self.gate3 = Prediction_Generator(c_list[0])

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Sequential(
            nn.Conv2d(c_list[0], num_classes, kernel_size=1),
        )

        # Contrast-Driven Feature Aggregation modules
        self.decouple = DecoupleLayer(c_list[5], 32)
        self.cdfa1 = ContrastDrivenFeatureAggregation(c_list[0], c_list[0], guide_dim=32, num_heads=4)
        self.cdfa2 = ContrastDrivenFeatureAggregation(c_list[1], c_list[1], guide_dim=32, num_heads=4)
        self.cdfa3 = ContrastDrivenFeatureAggregation(c_list[2], c_list[2], guide_dim=32, num_heads=4)
        self.cdfa4 = ContrastDrivenFeatureAggregation(c_list[3], c_list[3], guide_dim=32, num_heads=4)
        self.cdfa5 = ContrastDrivenFeatureAggregation(c_list[4], c_list[4], guide_dim=32, num_heads=4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            if n > 0:
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # 增加对 fan_out 为 0 的保护，防止除零错误 (ZeroDivisionError)
            if fan_out > 0:
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))#图像size减半
        t1 = out # b, 8, 128, 128

        out = self.encoder2(t1)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out # b, 16, 64, 64

        out = self.encoder3(t2)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out # b, 24, 32, 32

        out = self.encoder4(t3)
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2))
        t4 = out # b, 32, 16, 16

        out = self.encoder5(t4)
        out = F.gelu(F.max_pool2d(self.ebn5(out), 2))
        t5 = out # b, 48, 8, 8

        out = self.encoder6(t5)
        out = F.gelu(out) # b, 64, 8, 8

        # Decouple features for guidance
        f_fg, f_bg= self.decouple(out)

        out = self.decoder1(out)
        out = F.gelu(self.dbn1(out)) # b, 48, 8, 8

        t5_ref = self.cdfa5(t5, f_fg, f_bg)
        out, gt_pre5 = self.pred1(out)
        out = self.merge5(out, t5_ref, gt_pre5) # b, 48, 8, 8
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)


        out = self.decoder2(out)
        out = F.gelu(F.interpolate(self.dbn2(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 32, 16, 16
        t4_ref = self.cdfa4(t4, f_fg, f_bg)
        out, gt_pre4 = self.pred2(out)
        out = self.merge4(out, t4_ref, gt_pre4) # b, 32, 16, 16
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)

        out = self.decoder3(out)
        out = F.gelu(F.interpolate(self.dbn3(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 24, 32, 32
        t3_ref = self.cdfa3(t3, f_fg, f_bg)
        out, gt_pre3, weight1 = self.gate1(out)
        out = self.merge3(out, t3_ref, gt_pre3, weight1) # b, 24, 32, 32
        weight1 = F.interpolate(weight1, scale_factor=8, mode ='bilinear', align_corners=True)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)

        out = self.decoder4(out)
        out = F.gelu(F.interpolate(self.dbn4(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 16, 64, 64
        t2_ref = self.cdfa2(t2, f_fg, f_bg)
        out, gt_pre2, weight2 = self.gate2(out)
        out = self.merge2(out, t2_ref, gt_pre2, weight2) # b, 16, 64, 64
        weight2 = F.interpolate(weight2, scale_factor=4, mode ='bilinear', align_corners=True)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)

        out = self.decoder5(out)
        out = F.gelu(F.interpolate(self.dbn5(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 8, 128, 128
        t1_ref = self.cdfa1(t1, f_fg, f_bg)
        out, gt_pre1, weight3 = self.gate3(out)
        out = self.merge1(out, t1_ref, gt_pre1, weight3)
        weight3 = F.interpolate(weight3, scale_factor=2, mode ='bilinear', align_corners=True)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)

        out = self.final(out)
        out = F.interpolate(out,scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W

        gt_pre1 = torch.sigmoid(gt_pre1)
        gt_pre2 = torch.sigmoid(gt_pre2)
        gt_pre3 = torch.sigmoid(gt_pre3)
        gt_pre4 = torch.sigmoid(gt_pre4)
        gt_pre5 = torch.sigmoid(gt_pre5)

        # final_out = (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), (weight1, weight2, weight3), torch.sigmoid(out)
        return torch.sigmoid(out)