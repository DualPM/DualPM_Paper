import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud
from PIL import Image

import dualpm_paper.dataset as dd
from dualpm_paper.utils import rescale_im_and_mask

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

    tensor = np.empty(shape, dtype=values.dtype)
    mask = np.zeros(shape[:-1], dtype=np.bool8)

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
    pointmap, points_mask = _read_coo_npz(path)
    pointmap = torch.from_numpy(pointmap)
    pointmap = F.pad(pointmap, (0, 1), "constant", 0)
    pointmap[..., -1] = points_mask
    return pointmap


def _read_feats_npz(path: Path) -> torch.Tensor:
    feats, feats_mask = (torch.from_numpy(f) for f in _read_coo_npz(path))

    if not feats.is_integer():
        return feats

    feats = feats.float() / 127 - 1
    return feats


class InternetPointmapDataset(tud.Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        num_layers: int,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.resolution = image_size
        self.num_layers = num_layers

        self._points_dir = self.root / f"pointmaps_{self.resolution}"
        if not self._points_dir.exists():
            raise FileNotFoundError(
                f"Points directory {self._points_dir} does not exist"
            )

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

        ids = (p.stem for p in self._points_dir.glob("*.npz"))

        if include_ids or exclude_ids:
            ids = set(ids)
        if include_ids:
            ids &= set(include_ids)
        if exclude_ids:
            ids -= set(exclude_ids)

        self.ids = sorted(ids)
        self.collate_fn = dd.PointmapDataset.collate_fn

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]

        # load the pointmapå
        pointmap = _read_pointmap_npz(self._points_dir / f"{id_}.npz")[
            :, :, : self.num_layers
        ]

        feats, feats_mask = None, None
        rgb_image = None
        input_image = None

        if self._feats_dir is not None:
            feats, feats_mask = _read_feats_npz(self._feats_dir / f"{id_}.npz")
            input_image = feats

        else:
            rgb_image = torch.from_numpy(
                np.array(Image.open(self._render_dir / f"{id_}_rgb.png"))
            )

            input_image = rgb_image

        mask = torch.from_numpy(
            np.array(Image.open(self._mask_dir / f"{id_}_mask.png"))
        )

        input_image, mask = rescale_im_and_mask(
            input_image, mask, (self.resolution, self.resolution)
        )

        return input_image, pointmap, mask, id_
