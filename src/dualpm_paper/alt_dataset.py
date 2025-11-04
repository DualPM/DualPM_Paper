import logging
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud
from PIL import Image

import dualpm_paper.dataset as dd
from dualpm_paper.utils import (
    rescale_im_and_mask,
    GLTF2,
    process_gltf,
    OneMeshGltf,
    read_meta,
    read_camera,
    read_fuse_image,
    read_mask,
    read_image,
)
from dualpm_paper.skin import quaternion_to_matrix
from dualpm_paper.skin import skin_mesh

logger = logging.getLogger(__name__)


def _read_coo_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a dense tensor, and matching mask for a given sparse COO npz file.
    contains keys: shape tuple[int, ...], indices (N,len(shape)), values (N, C)

    returns:
        tensor: (*shape)
        mask: (*shape[:-1])
    """

    data = np.load(path)
    shape = data["shape"]
    indices = data["indices"]
    values = data["values"]

    tensor = np.zeros(shape, dtype=values.dtype)
    mask = np.zeros(shape[:-1], dtype=np.bool_)

    if indices.shape[-1] == 3:
        tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    elif indices.shape[-1] == 2:
        tensor[indices[:, 0], indices[:, 1]] = values
        mask[indices[:, 0], indices[:, 1]] = True
    else:
        raise ValueError(f"Invalid indices shape: {indices.shape}")

    return tensor, mask


def _read_pointmap_npz(path: Path) -> torch.Tensor:
    pointmap, points_mask = (torch.from_numpy(t) for t in _read_coo_npz(path))
    return pointmap


def _read_feats_npz(path: Path) -> torch.Tensor:
    feats, feats_mask = (torch.from_numpy(f) for f in _read_coo_npz(path))

    if feats.is_floating_point():
        return feats

    feats = feats.float() / 127 - 1
    return feats


class DualPmDataset(tud.Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        num_layers: int,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        exclude_5000s: bool = True,
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.resolution = image_size
        self.image_size = (image_size, image_size)
        self.num_layers = num_layers

        self._feats_dir = self.root / "features"
        if not self._feats_dir.exists():
            self._feats_dir = None
            logger.warning(f"Features directory {self._feats_dir} does not exist")

        self._mask_dir = self.root / "masks"
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Masks directory {self._mask_dir} does not exist")

        self._render_dir = self.root / "renders"
        if not self._render_dir.exists():
            raise FileNotFoundError(
                f"Renders directory {self._render_dir} does not exist"
            )

        # exclude 5000s as they are corrupted..
        ids = (
            p.stem.split("_rgb")[0]
            for p in self._render_dir.glob("*_rgb.png")
            if not exclude_5000s or not p.stem.isnumeric() or int(p.stem) % 5000
        )

        if include_ids or exclude_ids:
            ids = set(ids)
        if include_ids:
            ids &= set(include_ids)
        if exclude_ids:
            ids -= set(exclude_ids)

        self.ids = sorted(ids)
        self.collate_fn = dd.PointmapDataset.collate_fn

    def _read_images(self, file_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self._feats_dir is not None:
            feats = read_fuse_image(self._feats_dir / f"{file_id}_feat.png")
            input_image = feats

        else:
            rgb_image = torch.from_numpy(
                np.array(Image.open(self._render_dir / f"{file_id}_rgb.png"))
            )

            input_image = rgb_image

        mask = torch.from_numpy(
            np.array(Image.open(self._mask_dir / f"{file_id}_mask.png"))
        )
        input_image, mask = rescale_im_and_mask(
            input_image, mask, (self.resolution, self.resolution)
        )
        return input_image, mask

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        """Combines mulitple PointmapBatch of different examples into a single PointmapBatch"""
        return dd.PointmapDataset.collate_fn(batch)


class RasteredDataset(DualPmDataset):
    """
    dataset to be used if you pre-raster the pointmaps
    using scripts/raster_pointmaps.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._points_dir = self.root / f"pointmaps_{self.resolution}"
        if not self._points_dir.exists():
            raise FileNotFoundError(
                f"Points directory {self._points_dir} does not exist"
            )
        self.render_at_load = False

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]

        # load the pointmap
        pointmap = _read_pointmap_npz(self._points_dir / f"{id_}.npz")[
            :, :, : self.num_layers
        ]

        # match the expected shape of the rest of the code.. (NC, H, W)
        pointmap = einops.rearrange(pointmap, "h w n c -> (n c) h w")

        feats, feats_mask = None, None
        rgb_image = None
        input_image = None

        input_image, mask = self._read_images(id_)

        return input_image, pointmap, mask, id_

    def __len__(self) -> int:
        return len(self.ids)


def _read_shape(path: Path) -> OneMeshGltf:
    gltf = GLTF2().load(path)
    return process_gltf(gltf)


def _transpose(x: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(x, "... c r -> ... r c")


def _read_pose(path: Path) -> torch.Tensor:
    """
    read (num joints, 7)
    returns (num joints, 4, 4) as col major transforms
    """

    data = np.load(path)
    quat, pos = torch.from_numpy(data["poses"]).split([4, 3], dim=-1)
    rotmat = quaternion_to_matrix(quat)

    transform = torch.zeros(
        *quat.shape[:-1], 4, 4, device=quat.device, dtype=quat.dtype
    )
    transform[..., :3, :3] = rotmat
    transform[..., :3, 3] = pos
    transform[..., 3, 3] = 1
    return transform


class RasterizeDataset(DualPmDataset):
    """
    standard dataset, returns models to be rendered
    """

    def __init__(self, root: str | Path, image_size: int, num_layers: int, **kwargs):
        super().__init__(root, image_size, num_layers)
        self.shape_root = self.root / "shapes"
        self.shapes = {
            p.name: _read_shape(p / f"{p.stem}_shape.gltf")
            for p in self.shape_root.iterdir()
        }
        self.render_at_load = kwargs.get("render_at_load", False)

    def apply_pose(self, pose: torch.Tensor, shape: OneMeshGltf) -> OneMeshGltf:
        shape.local_joint_transforms = _transpose(pose)

        verts, global_joint_transforms, *_ = skin_mesh(shape)
        return verts, global_joint_transforms

    def _get_render_args(self, file_id: str) -> dict:
        pose = _read_pose(self.root / "poses" / f"{file_id}_pose.npz")
        meta = read_meta(self.root / "metadata" / f"{file_id}_metadata.txt")

        shape_id = meta["model_name"]
        focal_length = torch.tensor(meta["focal_length"], dtype=torch.float32)

        model = self.shapes[shape_id]
        view_matrix, *_ = read_camera(
            (self.root / "cameras" / f"{file_id}_camera.txt").open().read()
        )
        faces = model.faces

        view_verts, global_joint_transforms = self.apply_pose(pose, model)

        return (
            view_verts.to(torch.float32),
            model.vertices.to(torch.float32),
            faces,
            view_matrix,
            focal_length,
        )

    def __getitem__(self, idx: int):
        file_id = self.ids[idx]

        view_verts, canonical_verts, faces, view_matrix, focal_length = (
            self._get_render_args(file_id)
        )

        render_args = dict(
            pose_verts=view_verts,
            canonical_verts=canonical_verts,
            faces=faces,
            model_view=view_matrix,
            focal_length=focal_length,
        )

        model_targets = None
        if self.render_at_load:
            model_targets = einops.rearrange(
                self.renderer(**{k: v[None] for k, v in render_args.items()}),
                "b h w n c-> b (n c) h w",
            )

            render_args = None

        input_image, mask = self._read_images(file_id)

        return (
            input_image,
            model_targets,
            mask,
            file_id,
            render_args,
        )

    def __len__(self) -> int:
        return len(self.ids)


class TestDataset(tud.Dataset):
    def __init__(
        self,
        image_dir: Path | None,
        mask_dir: Path,
        feat_dir: Path,
        image_size: tuple[int, int],
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
    ):
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.mask_dir = Path(mask_dir)
        self.feat_dir = Path(feat_dir) if feat_dir is not None else None

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.ids = self._find_ids(include_ids, exclude_ids)

    def _find_ids(
        self, include_ids: list[str] | None = None, exclude_ids: list[str] | None = None
    ):
        image_ids = None
        feat_ids = None
        if self.image_dir is not None:
            image_ids = set(
                p.stem.split("_rgb")[0] for p in self.image_dir.glob("*_rgb.png")
            )
        if self.feat_dir is not None:
            feat_ids = set(
                p.stem.split("_feat")[0] for p in self.feat_dir.glob("*_feat.png")
            )
        ids = set(p.stem.split("_mask")[0] for p in self.mask_dir.glob("*_mask.png"))

        if image_ids is not None:
            ids &= image_ids
        if feat_ids is not None:
            ids &= feat_ids
        if include_ids:
            ids &= set(include_ids)
        if exclude_ids:
            ids -= set(exclude_ids)

        return sorted(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        file_id = self.ids[idx]
        image = None
        if self.image_dir is not None:
            image = read_image(
                (self.image_dir / f"{file_id}_rgb.png").open("rb").read(),
            )

        feat = read_fuse_image(
            (self.feat_dir / f"{file_id}_feat.png").open("rb").read()
        )
        mask = read_mask((self.mask_dir / f"{file_id}_mask.png").open("rb").read())

        feat, mask = rescale_im_and_mask(feat, mask, self.image_size)

        return file_id, image, mask, feat

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        file_ids, images, masks, feats = list(zip(*batch, strict=True))
        images = torch.stack(images) if images[0] is not None else None
        masks, feats = torch.stack(masks), torch.stack(feats)
        return file_ids, images, masks, feats
