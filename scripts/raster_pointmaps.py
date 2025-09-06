"""
rasterize pointmaps from dataset and save to sparse npz file
"""

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.utils.data as tud
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dualpm_paper.dataset import MeshToDualPointmap, PointmapBatch


def _extract_sparse(pointmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pointmap_values = pointmap[..., :7]

    shape_ = pointmap_values.shape

    indices = pointmap_values[..., 6].nonzero()

    pointmap_values = pointmap_values[indices[:, 0], indices[:, 1], indices[:, 2]]
    index_dtype = torch.uint8 if max(shape_[:-1]) < 255 else torch.uint16

    return pointmap_values, indices.to(index_dtype), tuple(shape_)


def raster_pointmaps(cfg: DictConfig):
    output_path = Path(cfg.dataset_root) / f"pointmaps_{cfg.resolution}"
    completed_ids = set(p.stem for p in output_path.glob("*.npz"))

    dataset = hydra.utils.instantiate(cfg.dataset, exclude_ids=completed_ids)
    loader = tud.DataLoader(
        dataset,
        batch_size=12,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset.collate_fn,
    )

    # rasterize at least 8 layers even if the dataset only has less
    renderer = MeshToDualPointmap(
        im_size=dataset.image_size,
        num_layers=max(dataset.num_layers, 8),
        return_on_cpu=True,
        scale_targets=False,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader):
        batch = PointmapBatch(*batch)
        pointmap = renderer(**batch.render_args)
        for id_, pointmap_k in zip(batch.data_id, pointmap, strict=True):
            pointmap_values, indices, shape_ = _extract_sparse(pointmap_k)
            np.savez(
                output_path / f"{id_}.npz",
                values=pointmap_values.numpy(),
                indices=indices.numpy(),
                shape=tuple(shape_),
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise ValueError("Usage: python raster_pointmaps.py <config_path>")

    cfg_path = Path(sys.argv[1]).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config at {cfg_path} does not exist")

    cfg = OmegaConf.load(cfg_path)
    raster_pointmaps(cfg)
