"""
Copyright 2025 University of Oxford
Author: Ben Kaye
Licence: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Literal

import einops
import torch
import torch.nn as nn
from torch import Tensor

from dualpm_paper.backbones import SongUNet


class ConvUnet(nn.Module):
    """End to end DualPM predictor"""

    num_layers: int
    channels_per_layer: int
    resolution: int

    # capture out_channels,img_resolution to override
    def __init__(
        self,
        num_layers: int,
        channels_per_layer: Literal[6, 7, 8],
        resolution: int,
        out_channels: int | None = None,
        img_resolution: int | None = None,
        **unet_args,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.channels_per_layer = channels_per_layer
        self.resolution = resolution

        out_channels = channels_per_layer * num_layers

        self.net = SongUNet(
            img_resolution=resolution,
            out_channels=out_channels,
            **unet_args,
        )

    def forward(self, input_image: Tensor) -> Tensor:
        x = einops.rearrange(input_image, "b h w c-> b c h w")
        return self.net(x)


class SequentialUnet(nn.Module):
    """Multi-stage DualPM predictor. That predicts and then conditions the second stage on the canonical or depth map (+ features)"""

    num_layers: int
    channels_per_layer: int
    resolution: int

    def __init__(
        self,
        in_channels: int,
        resolution: int,
        model_channels: int = 128,
        num_layers: int = 1,
        channels_per_layer: Literal[6, 7, 8] = 6,
        depth_first: bool = False,
        detach_conditioning: bool = True,
        image_conditioning_stage2: bool = False,
        model_overrides: dict | None = None,
    ):
        """
        Args:
            input_channels: number of channels in the input image
            resolution: resolution of the input image (square)
            model_channels: number of channels in the model
            num_layers: number of layers in the pointmap
            channels_per_layer: number of channels per layer in the pointmap
            depth_first: if True, depth unet is first
            detach_conditioning: if True, stop gradient of input to second stage
            image_conditioning_stage2: if True, condition the second stage on the image and the output of the first stage
            model_overrides: Override kwargs for SongUNet
        """

        super().__init__()
        self.depth_first = depth_first
        self.channels_per_layer = channels_per_layer
        self.image_conditioning_stage2 = image_conditioning_stage2
        self.detach_conditioning = detach_conditioning
        self.num_layers = num_layers
        self.resolution = resolution
        self.channels_per_layer = channels_per_layer

        assert num_layers == 1 or num_layers > 1 and channels_per_layer >= 7, (
            "Multi-layer pointmaps must have at least 7 channels per layer"
        )

        predictions = channels_per_layer * num_layers

        canon_encoder_args = dict(
            in_channels=in_channels,
            img_resolution=resolution,
            model_channels=model_channels,
            out_channels=3,
        )
        depth_encoder_args = dict(
            in_channels=in_channels,
            img_resolution=resolution,
            model_channels=model_channels,
            out_channels=3,
        )

        if model_overrides is not None:
            canon_encoder_args |= model_overrides
            depth_encoder_args |= model_overrides

        if depth_first:
            canon_encoder_args["out_channels"] = predictions - 3
            canon_encoder_args["in_channels"] = 3
            if self.image_conditioning_stage2:
                canon_encoder_args["in_channels"] += in_channels
        else:
            depth_encoder_args["out_channels"] = predictions - 3
            depth_encoder_args["in_channels"] = 3
            if self.image_conditioning_stage2:
                depth_encoder_args["in_channels"] += in_channels

        self.nets = nn.ModuleDict(
            dict(
                depth_encoder=SongUNet(**depth_encoder_args),
                canonical_encoder=SongUNet(**canon_encoder_args),
            )
        )

    def forward(self, feats_in: Tensor) -> Tensor:
        """input_image: (B, H, W, Cin)

        Returns:
            output: (B, Cout, H, W)

        Output is computed in 2 stages:
        - First stage: Unet of that predicts either first layer canonical or depth map
        - Second stage: Unet of missing parameters
        """

        feats_in = einops.rearrange(feats_in, "b h w c-> b c h w")
        vis_depth, vis_canon = None, None
        stage2_input = []
        if self.image_conditioning_stage2:
            stage2_input += [feats_in]

        if self.depth_first:
            vis_depth = self.nets.depth_encoder(feats_in)

            stage2_input += [
                vis_depth.detach() if self.detach_conditioning else vis_depth
            ]
        else:
            vis_canon = self.nets.canonical_encoder(feats_in)
            stage2_input += [
                vis_canon.detach() if self.detach_conditioning else vis_canon
            ]

        stage2_input = torch.cat(stage2_input, dim=1)

        if self.depth_first:
            stage2_predictions = self.nets.canonical_encoder(stage2_input)
            vis_canon = stage2_predictions[:, :3]
        else:
            stage2_predictions = self.nets.depth_encoder(stage2_input)
            vis_depth = stage2_predictions[:, :3]

        if self.num_layers == 1:
            output = [vis_canon, vis_depth]
            if self.channels_per_layer > 6:
                confidence = stage2_predictions[:, [3]]
                output += [confidence]
            return torch.cat(output, dim=1)

        # we have already extracted the first 3 outputs
        channel_offset = self.channels_per_layer - 3
        occupancy_and_confidence = stage2_predictions[:, 3:channel_offset]

        # construct first layer
        layer1 = torch.cat([vis_canon, vis_depth, occupancy_and_confidence], dim=1)

        output = torch.cat(
            [layer1, stage2_predictions[:, channel_offset:, :, :]], dim=1
        )

        return output
