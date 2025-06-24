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

import logging
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Literal

import nvdiffrast.torch as drt
import torch
import torch.utils.data as tud
from dualpm import render_dual_point_map
from dualpm.utils import perspective_matrix, tan_half_fov
from einops import rearrange, einsum
from torch import Tensor

from dualpm_paper.skin import SkinnedMesh
from dualpm_paper.utils import (
    BatchDataclass,
    read_camera,
    read_fuse_image,
    read_gltf,
    read_image,
    read_mask,
    read_meta,
    read_npy,
    rescale_im_and_mask,
)

logger = logging.getLogger(__name__)


class FileSuffixes(Enum):
    RGB = "_rgb.png"
    MASK = "_mask.png"
    SHAPE = "_shape.gltf"
    CAMERA = "_camera.txt"
    METADATA = "_metadata.txt"
    DINO = "_feat.npy"
    SD_DINO = "_feat.png"


@dataclass
class PointmapBatch(BatchDataclass):
    input_image: Tensor
    model_targets: Tensor
    mask: Tensor
    data_id: str
    render_args: dict | None = None
    vertices: Tensor | None = None
    faces: Tensor | None = None


MALE_HORSE_EXTENTS = (0.68662, 2.26969, 2.71548)


@dataclass
class PointmapBatch(BatchDataclass):
    input_image: Tensor
    model_targets: Tensor
    mask: Tensor
    data_id: str
    render_args: dict | None = None
    vertices: Tensor | None = None
    faces: Tensor | None = None


class MeshToDualPointmap:
    image_size: tuple[int, int]
    num_layers: int
    sensor_height: float
    context: drt.RasterizeGLContext | None

    def __init__(
        self,
        im_size: tuple[int],
        num_layers: int,
        return_on_cpu: bool,
        subtract_depth: bool = True,
        sensor_height: float = 36.0,
        scale_targets: bool = False,
    ):
        self.image_size = tuple(im_size)
        self.num_layers = num_layers
        self.sensor_height = sensor_height
        self.subtract_depth = subtract_depth
        self.return_on_cpu = return_on_cpu

        self.context = drt.RasterizeGLContext(output_db=False)

        self.raster_func = partial(
            render_dual_point_map,
            resolution=self.image_size,
            num_layers=self.num_layers,
            context=self.context,
            subtract_depth=self.subtract_depth,
            return_on_cpu=self.return_on_cpu,
        )
        self.scale_targets = scale_targets

        if self.scale_targets:
            self._extent_scale = torch.tensor(MALE_HORSE_EXTENTS).prod(dim=-1)

    def calculate_dual_pointmap(
        self,
        pose_verts: torch.Tensor,
        canonical_verts: torch.Tensor,
        faces: torch.Tensor,
        model_view: torch.Tensor,
        focal_length: torch.Tensor,
    ):
        # FIXME move this to the collate_fn
        if isinstance(focal_length, list):
            focal_length = torch.stack(focal_length)
        if isinstance(model_view, list):
            model_view = torch.stack(model_view)

        faces = [f.type(torch.int32) for f in faces]

        _tan_half_fov = tan_half_fov(focal_length, self.sensor_height)
        projection = perspective_matrix(_tan_half_fov)

        with torch.no_grad():
            dual_pointmap = self.raster_func(
                canonical_vertices=canonical_verts,
                reconstruction_vertices=pose_verts,
                faces=faces,
                model_view=model_view,
                projection=projection,
            )

        if self.scale_targets:
            # Note this NAIVE scaling is not scaling about the object center of mass
            # if subtract_z is not used this would be bad!

            # scale by the ratio of bounding box volumes
            axis_lens = torch.stack(
                [c.max(dim=0).values - c.min(dim=0).values for c in canonical_verts]
            )
            scale_factors = (self._extent_scale / axis_lens.prod(dim=-1)).pow(1 / 3)

            dual_pointmap = einsum(dual_pointmap, scale_factors, "b ..., b -> b ...")

        return dual_pointmap

    def __call__(self, pose_verts, canonical_verts, faces, model_view, focal_length):
        return self.calculate_dual_pointmap(
            pose_verts, canonical_verts, faces, model_view, focal_length
        )


class PointmapDataset(tud.Dataset):
    root: Path
    feat_root: Path
    renderer: MeshToDualPointmap | None
    image_size: tuple[int, int]
    num_layers: int
    dino_features: bool
    render_at_load: bool
    scale_targets: bool
    input_mode: Literal["dino", "sd_dino", "rgb"]
    sensor_height: float = 36.0

    def __init__(
        self,
        root: str,
        feat_root: str | None,
        image_size: tuple[int, int],
        num_layers: int,
        input_mode: Literal["dino", "sd_dino", "rgb"],
        render_at_load: bool = False,
        scale_targets: bool = False,
        exclude_ids: list[str] | None = None,
        include_ids: list[str] | None = None,
        exclude_5000s: bool = True,
    ):
        """Dataset for loading and processing mesh data with simple rendering.

        Args:
            root: Path to root directory containing mesh files and metadata
            fuse_root: Path to directory containing feature images
            image_size: Target size for rendered images (H, W)
            num_layers: Number of depth layers to render
            extract_features: If True, use RGB images instead of feature images
            render_at_load: If True, initialize renderer during loading
        """
        self.root = Path(root)
        self.feat_root = Path(feat_root) if feat_root is not None else None

        if isinstance(image_size, list):
            image_size = tuple(image_size)
        elif not isinstance(image_size, tuple):
            image_size = (image_size, image_size)

        self.image_size = image_size
        self.num_layers = num_layers
        self.dino_features = input_mode
        self.render_at_load = render_at_load
        self.scale_targets = scale_targets
        self.input_mode = input_mode

        self.renderer = (
            MeshToDualPointmap(
                im_size=self.image_size,
                num_layers=self.num_layers,
                return_on_cpu=True,
                scale_targets=self.scale_targets,
            )
            if render_at_load
            else None
        )

        # Map file IDs
        self.file_ids = self._map_files(include_ids, exclude_ids, exclude_5000s)

    def _map_files(
        self,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        exclude_5000s: bool = True,
    ) -> list[str]:
        """Create mapping of valid file IDs"""
        # Find all files with required extensions

        match self.input_mode:
            case "dino":
                ids = set(
                    f.name.split(FileSuffixes.DINO.value)[0]
                    for f in self.feat_root.glob(f"*{FileSuffixes.DINO.value}")
                )
            case "sd_dino":
                ids = set(
                    f.name.split(FileSuffixes.SD_DINO.value)[0]
                    for f in self.feat_root.glob(f"*{FileSuffixes.SD_DINO.value}")
                )
            case "rgb":
                ids = set(
                    f.name.split(FileSuffixes.RGB.value)[0]
                    for f in self.root.glob(f"*{FileSuffixes.RGB.value}")
                )

        if exclude_ids:
            ids -= set(exclude_ids)

        if include_ids:
            ids &= set(include_ids)

        ids = sorted(ids)

        if exclude_5000s:
            ids = [id_ for id_ in ids if not id_.isnumeric() or int(id_) % 5000]

        return ids

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, str, dict | None]:
        """returns the components of PointmapBatch as a tuple"""
        file_id = self.file_ids[idx]

        skinned_mesh: SkinnedMesh = read_gltf(
            self.root / f"{file_id}{FileSuffixes.SHAPE.value}"
        )

        # Load camera parameters
        view_matrix, *_ = read_camera(
            (self.root / f"{file_id}{FileSuffixes.CAMERA.value}").open().read()
        )
        metadata = read_meta(self.root / f"{file_id}{FileSuffixes.METADATA.value}")

        # Load images

        match self.input_mode:
            case "dino":
                input_image = read_npy(
                    self.feat_root / f"{file_id}{FileSuffixes.DINO.value}"
                )
            case "sd_dino":
                input_image = read_fuse_image(
                    self.feat_root / f"{file_id}{FileSuffixes.SD_DINO.value}"
                )
            case "rgb":
                input_image = read_image(
                    self.root / f"{file_id}{FileSuffixes.RGB.value}"
                )

        mask = read_mask(self.root / f"{file_id}{FileSuffixes.MASK.value}")

        # Resize images
        input_image, mask = rescale_im_and_mask(input_image, mask, self.image_size)

        # Get camera parameters
        focal_length = torch.tensor(metadata["focal_length"], dtype=torch.float32)
        # Generate model targets
        model_targets = None

        render_args = dict(
            pose_verts=skinned_mesh.posed_vertices,  # Add batch dimension
            canonical_verts=skinned_mesh.vertices,
            faces=skinned_mesh.faces,
            model_view=view_matrix,
            focal_length=focal_length,
        )

        # render on the dataloader (not ideal as requires GPU and GPU memory)
        if self.render_at_load:
            model_targets = rearrange(
                self.renderer(**{k: v[None] for k, v in render_args.items()}),
                "b h w n c-> b (n c) h w",
            )

            render_args = None

        return (
            input_image,
            model_targets,
            mask,
            file_id,
            render_args,
        )

    @staticmethod
    def collate_fn(input: list[tuple]) -> tuple:
        """Combines mulitple PointmapBatch of different examples into a single PointmapBatch"""
        keys = tuple(f.name for f in dataclass_fields(PointmapBatch))
        result = []
        for k, v in zip(keys, zip(*input, strict=False), strict=False):
            match k:
                case "data_id":
                    result.append(v)
                case "render_args":
                    result.append({k: [d[k] for d in v] for k in v[0]})
                case "local_bones_transforms":
                    result.append(v)
                case "global_bone_transforms":
                    result.append(v)
                case _:
                    if all(i is None for i in v):
                        result.append(None)
                    else:
                        result.append(torch.stack(v, dim=0))

        return tuple(result)


def extract_ids_subset(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).open().readlines()]
