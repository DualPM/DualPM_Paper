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

import io
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL.Image import Image, open as image_open
from pygltflib import GLTF2
from torch import Tensor

from dualpm_paper.skin import (
    OneMeshGltf,
    SkinnedMesh,
    skin_mesh,
)
from dualpm_paper.skin import (
    extract_gltf_data as process_gltf,
)


def iter_tensors(obj):
    for name in dir(obj):
        if "__" in name:
            continue

        attr = getattr(obj, name)
        if isinstance(attr, Tensor):
            yield name, attr


@dataclass
class BoneParameterisation:
    parents: list
    lengths: Tensor


@dataclass
class BoneMeta:
    n_bones: int
    posed_vertices: Tensor
    skinning_matrices: Tensor
    skingltf_mesh: OneMeshGltf


class BatchDataclass:
    def __iter__(self):
        for field in dataclass_fields(self):
            yield getattr(self, field.name)

    def to(self, device) -> None:
        """copy tensors to device and update ref"""
        for name, tensor in iter_tensors(self):
            setattr(self, name, tensor.to(device))

    def n_batches(self, n=1) -> None:
        """retains first n batches"""
        for name, tensor in iter_tensors(self):
            setattr(self, name, tensor[:n, ...])

    def drop_batch(self) -> None:
        """discard all but first batch and remove batch dim"""
        self.n_batches(n=1)
        for name, tensor in iter_tensors(self):
            setattr(self, name, tensor[0, ...])

    def __repr__(self) -> str:
        name, first_tensor = next(iter_tensors(self))

        static_string = (
            f"batch: {first_tensor.shape[0]}\ndevice: {first_tensor.device}\n"
        )
        shapes_string = "\n".join(
            [
                f"{name}: {str(tuple(tensor.shape[1:]))}"
                for name, tensor in iter_tensors(self)
            ]
        )
        return f"{static_string}{shapes_string}"

    def __str__(self) -> str:
        return self.__repr__()


def dataclass_to(dataclass, device):
    for name, tensor in iter_tensors(dataclass):
        setattr(dataclass, name, tensor.to(device))


def get_root_bone_pos(src_bone_data: Tensor, root_index: int = 0) -> Tensor:
    """Extract root bone position from bone data.

    Args:
        src_bone_data: Bone transform matrices
        root_index: Index of root bone (default: 0)

    Returns:
        Root bone position vector
    """
    return src_bone_data[root_index, 3, :3]


def transform(matrix: Tensor, vector: Tensor) -> Tensor:
    """Apply transformation matrix to vector(s).

    Args:
        matrix: 3x4 or 4x4 transformation matrix
        vector: (..., 3) vector(s) to transform

    Returns:
        Transformed vector(s) with same shape as input
    """
    return (matrix[:3, :3] @ vector[..., None])[..., 0] + matrix[:3, 3]


def get_height(verts: Tensor, axis: int = 1) -> Tensor:
    """Calculate height of vertex set along specified axis.

    Args:
        verts: Vertex positions
        axis: Axis along which to measure height (default: 1)

    Returns:
        Height measurement as scalar tensor
    """
    min_ = verts.min(dim=0).values
    max_ = verts.max(dim=0).values
    return max_[axis] - min_[axis]


def rescale_im_and_mask(
    im: Tensor, mask: Tensor, im_shape: tuple, mask_threshold: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Rescale image and mask to target shape.

    Args:
        im: H W C
        mask: H W
        im_shape: (H, W) target shape
        mask_threshold: threshold for mask binary

    Returns:
        Tuple of (rescaled masked image, rescaled binary mask)
    """
    assert isinstance(im_shape, tuple), "im_shape must be a tuple"
    assert mask.dim() == 2, "mask must be 2D!"
    assert mask.max() <= 1.0, "mask must be in range [0,1]"

    # Add 2 dims to mask, drop 1
    mask = F.interpolate(mask[None, None, :, :], im_shape, mode="nearest")[0]

    mask[mask >= mask_threshold] = 1.0
    mask[mask < mask_threshold] = 0.0

    # Add then drop batch dim
    im = rearrange(im, "h w c -> c h w")
    im = F.interpolate(im[None], im_shape, mode="bilinear")[0]

    masked_im = torch.mul(mask, im)
    masked_im = rearrange(masked_im, "c h w -> h w c")

    mask = mask[0]  # drop channel dim

    return masked_im, mask


def change_convention(camera_pose: Tensor, convention=None) -> tuple[Tensor, Tensor]:
    """Change coordinate convention for camera pose."""
    z_up_to_y_up = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)

    convention = z_up_to_y_up if convention is None else convention

    # augment the rotation to (4, 4)
    transform = torch.block_diag(z_up_to_y_up, torch.tensor(1.0))

    # apply convention change
    camera_pose = transform @ camera_pose

    # invert pose to get extrinsic
    extrinsic = invert_transform(camera_pose)

    return extrinsic, camera_pose


def invert_transform(transform: Tensor) -> Tensor:
    """Invert a transformation matrix.

    transform: (..., 4, 4)

    returns inverse transform: (..., 4, 4)
    """
    assert transform.shape[-2:] == (4, 4)
    batch_mode = len(transform.shape) == 3
    assert transform.dim() <= 3

    inv = torch.zeros_like(transform, device=transform.device)
    inv[..., -1, -1] = 1.0

    rot = rearrange(transform[..., :-1, :-1], "... r c -> ... c r")
    t = -rot @ transform[..., :-1, [-1]]

    if batch_mode:
        inv[:, :-1, :-1] = rot
    else:
        inv[:-1, :-1] = rot
    inv[..., :-1, [-1]] = t

    return inv


def read_gltf(path: str, return_float32: bool = True) -> SkinnedMesh | None:
    """Read the GLTF file and return the Mesh posed with its provided skeleton

    Args:
        path: Path to the GLTF file
        return_float32: Whether to return the mesh in float32 or float64

    Returns:
        SkinnedMesh
    """
    try:
        gltf = GLTF2().load(path)
        mesh = process_gltf(gltf)

        posed_vertices, global_joint_transforms, skinning_matrices = skin_mesh(mesh)

        skinned_mesh = SkinnedMesh(
            *mesh, posed_vertices, global_joint_transforms, skinning_matrices
        )

        if return_float32:
            skinned_mesh = SkinnedMesh(
                *(
                    t.type(torch.float32)
                    if isinstance(t, Tensor) and t.dtype == torch.float64
                    else t
                    for t in skinned_mesh
                )
            )

        return skinned_mesh
    except Exception:
        return None


def read_camera(camera_text: str) -> tuple[Tensor, Tensor]:
    """Read camera parameters from text."""
    camera_pose = torch.tensor(
        [
            [float(x) for x in g.replace("\n", "").split(" ")]
            for g in camera_text.split("\n")
            if g != ""
        ]
    )

    view_matrix, camera_pose = change_convention(camera_pose)
    return view_matrix, camera_pose


def read_meta(path: str) -> dict | None:
    """Read metadata from YAML file."""
    try:
        path = Path(path)
        return yaml.safe_load(path.open())
    except Exception:
        return None


def read_npy(path: str) -> Tensor:
    """
    Read numpy array from file

    returns H W C tensor
    """
    return torch.tensor(np.load(path), dtype=torch.float32)


def read_fuse_image(fuse_image: bytes) -> Tensor:
    """Read and process fused image data."""
    img = torch.tensor(read_png(fuse_image), dtype=torch.float32) / 127 - 1

    # assume square
    h = img.shape[0]
    n_tiles = img.shape[1] // h

    # assume multiple of 16
    n_channels = 3 * n_tiles
    n_channels = 16 * (n_channels // 16)
    n_addon_channels = 3 - n_channels % 3

    feat = rearrange(img, "h (t w) c -> h w (t c)", t=n_tiles, c=3)
    feat = feat[:, :, :-n_addon_channels]
    return feat


def read_image(image: bytes) -> Tensor:
    """Read image data to tensor."""
    if isinstance(image, Image):
        image = np.array(image)
    else:
        image = read_png(image)
    return torch.tensor(image, dtype=torch.float32) / 255


def read_mask(mask: bytes) -> Tensor:
    """Read mask data to tensor."""
    mask = torch.tensor(np.array(read_png(mask)), dtype=torch.float32)
    if mask.max() > 1:
        mask /= 255
    return mask


def read_png(image: bytes) -> np.ndarray:
    """Read PNG image data."""
    try:
        return np.array(image_open(image))
    except ValueError:
        return np.array(image_open(io.BytesIO(image)))
