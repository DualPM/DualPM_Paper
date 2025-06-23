import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# U-Net implementation from EDM
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# modifed @ben-kaye 2025. change attention to use torch.nn.functional.scaled_dot_product_attention

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


def attn_old(q, k, v):
    w = AttentionOp.apply(q, k)
    a = torch.einsum("nqk,nck->ncq", w, v)
    return a


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **attn_kwargs
) -> torch.Tensor:
    """
    q: ((N B), C, L)
    k: ((N B), C, S)
    v: ((N B), C, S)

    returns ((N B), C, L)
    """
    q, k, v = (rearrange(t, "b c l -> b l c") for t in (q, k, v))
    a = F.scaled_dot_product_attention(q, k, v, **attn_kwargs)
    a = rearrange(a, "b l c -> b c l")
    return a


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=(1, 1),
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x, N_views_xa=1):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, N_views_xa=1):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, "b c -> (b c)")
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, "(b c) emb_ch -> b (c emb_ch)", b=b)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, "b c -> (b c)")
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, "(b c) emb_ch -> b (c emb_ch)", b=b)
        return x


class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, num_channels, num_heads=1, eps=1e-5):
        super().__init__()

        self.num_heads = num_heads
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)

        self.norm = GroupNorm(num_channels=num_channels, eps=eps)

        self.q_proj = Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel=1, **init_attn
        )
        self.kv_proj = Conv2d(
            in_channels=num_channels,
            out_channels=num_channels * 2,
            kernel=1,
            **init_attn,
        )

        self.out_proj = Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel=3, **init_zero
        )

    def forward(self, q, kv):
        q_proj = self.q_proj(self.norm(q)).reshape(
            q.shape[0] * self.num_heads, q.shape[1] // self.num_heads, -1
        )
        k_proj, v_proj = (
            self.kv_proj(self.norm(kv))
            .reshape(kv.shape[0] * self.num_heads, kv.shape[1] // self.num_heads, 2, -1)
            .unbind(2)
        )
        # a = attn_old(q_proj, k_proj, v_proj)
        a = attention(q_proj, k_proj, v_proj)
        x = self.out_proj(a.reshape(*q.shape)).add_(q)

        return x


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=(1, 1),
        resample_proj=False,
        adaptive_scale=True,
        init: dict | None = None,
        init_zero: dict | None = None,
        init_attn: dict | None = None,
    ):
        super().__init__()
        init = init or {}
        init_zero = init_zero or dict(init_weight=0)
        init_attn = init_attn or {}

        self.in_channels = in_channels
        self.out_channels = out_channels
        if emb_channels is not None:
            self.affine = Linear(
                in_features=emb_channels,
                out_features=out_channels * (2 if adaptive_scale else 1),
                **init,
            )
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb=None, N_views_xa=1):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = F.silu(self.norm1(x.add_(params)))

        x = F.silu(self.norm1(x))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            if N_views_xa != 1:
                B, C, H, W = x.shape
                # (B, C, H, W) -> (B/N, N, C, H, W) -> (B/N, N, H, W, C)
                x = x.reshape(B // N_views_xa, N_views_xa, *x.shape[1:]).permute(
                    0, 1, 3, 4, 2
                )
                # (B/N, N, H, W, C) -> (B/N, N*H, W, C) -> (B/N, C, N*H, W)
                x = x.reshape(
                    B // N_views_xa, N_views_xa * x.shape[2], *x.shape[3:]
                ).permute(0, 3, 1, 2)
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            # a = attn_old(q, k, v)
            a = attention(q, k, v)

            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
            if N_views_xa != 1:
                # (B/N, C, N*H, W) -> (B/N, N*H, W, C)
                x = x.permute(0, 2, 3, 1)
                # (B/N, N*H, W, C) -> (B/N, N, H, W, C) -> (B/N, N, C, H, W)
                x = x.reshape(B // N_views_xa, N_views_xa, H, W, C).permute(
                    0, 1, 4, 2, 3
                )
                # (B/N, N, C, H, W) -> # (B, C, H, W)
                x = x.reshape(B, C, H, W)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch
# taken from EDM repository https://github.com/NVlabs/edm/blob/main/training/networks.py#L372


class SongUNet(nn.Module):
    def __init__(
        self,
        # Image resolution at input/output.
        img_resolution,
        # Number of color channels at input.
        in_channels,
        # Number of color channels at output.
        out_channels,
        emb_dim_in=0,  # Input embedding dim.
        # Augmentation label dimensionality, 0 = no augmentation.
        augment_dim=0,
        # Base multiplier for the number of channels.
        model_channels=128,
        # Per-resolution multipliers for the number of channels.
        channel_mult=(1, 2, 2, 2),
        # Multiplier for the dimensionality of the embedding vector.
        channel_mult_emb=4,
        # Number of residual blocks per resolution.
        num_blocks=4,
        # List of resolutions with self-attention.
        attn_resolutions=(16,),
        # Dropout probability of intermediate activations.
        dropout=0.10,
        # Dropout probability of class labels for classifier-free guidance.
        label_dropout=0,
        # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        embedding_type="positional",
        # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        channel_mult_noise=0,
        # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        encoder_type="standard",
        # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        decoder_type="standard",
        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        resample_filter=(1, 1),
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        self.emb_dim_in = emb_dim_in
        if emb_dim_in > 0:
            emb_channels = model_channels * channel_mult_emb
        else:
            emb_channels = None
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Mapping.
        # self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        # self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        # self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        if emb_dim_in > 0:
            self.map_layer0 = Linear(
                in_features=emb_dim_in, out_features=emb_channels, **init
            )
            self.map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
            )

        if noise_channels > 0:
            self.noise_map_layer0 = Linear(
                in_features=noise_channels, out_features=emb_channels, **init
            )
            self.noise_map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
            )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    init_weight=0.2,
                    **init,
                )  # init_zero)

    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        emb = None

        if film_camera_emb is not None:
            if self.emb_dim_in != 1:
                film_camera_emb = (
                    film_camera_emb.reshape(film_camera_emb.shape[0], 2, -1)
                    .flip(1)
                    .reshape(*film_camera_emb.shape)
                )  # swap sin/cos
            film_camera_emb = F.silu(self.map_layer0(film_camera_emb))
            film_camera_emb = F.silu(self.map_layer1(film_camera_emb))
            emb = film_camera_emb

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux, N_views_xa)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux, N_views_xa)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux, N_views_xa)) / np.sqrt(2)
            else:
                x = (
                    block(x, emb=emb, N_views_xa=N_views_xa)
                    if isinstance(block, UNetBlock)
                    else block(x, N_views_xa=N_views_xa)
                )
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux, N_views_xa)
            elif "aux_norm" in name:
                tmp = block(x, N_views_xa)
            elif "aux_conv" in name:
                tmp = block(F.silu(tmp), N_views_xa)
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    # skip connection is pixel-aligned which is good for
                    # foreground features
                    # but it's not good for gradient flow and background features
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb=emb, N_views_xa=N_views_xa)
        return aux


class UpscalingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1
        )
        if in_channel != out_channel:
            self.conv_res = nn.Conv2d(
                in_channels=in_channel, out_channels=out_channel, kernel_size=1
            )
        else:
            self.conv_res = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.up(x)
        r = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x + self.conv_res(r))
        return x
