import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from dualpm_paper.pointmaps import PointmapModule
import trimesh


@hydra.main(config_path="../configs", config_name="infer", version_base="1.3")
def main(cfg: DictConfig):
    weights_path = Path(cfg.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    dataset = hydra.utils.instantiate(cfg.dataset)
    module: PointmapModule = hydra.utils.instantiate(cfg.module)

    # map to CPU
    state_dicts = torch.load(weights_path, map_location="cpu")
    if "model_state" in state_dicts:
        model_weight_dict = state_dicts["model_state"]
    else:
        model_weight_dict = state_dicts

    module.model.load_state_dict(
        model_weight_dict,
        strict=False,
    )

    module.model.to(cfg.device)
    module.model.eval()
    module.device = cfg.device

    loader = hydra.utils.instantiate(
        cfg.dataloader, dataset=dataset, collate_fn=dataset.collate_fn
    )

    def _loop():
        for batch in loader:
            file_ids, images, masks, feats = batch
            with torch.inference_mode():
                canon, posed, seq_mask = module.predict(
                    feats,
                    masks,
                    device=cfg.device,
                    confidence_threshold=cfg.confidence_threshold,
                )
            canon, posed, seq_mask = (t.clone().cpu() for t in (canon, posed, seq_mask))
            yield from zip(file_ids, canon, posed, seq_mask, strict=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_id, canon, posed, seq_mask in _loop():
        canon_pointcloud = canon[seq_mask]
        reconstruction_pointcloud = posed[seq_mask]

        trimesh.PointCloud(vertices=canon_pointcloud).export(
            output_dir / f"{file_id}_canon.ply"
        )
        trimesh.PointCloud(vertices=reconstruction_pointcloud).export(
            output_dir / f"{file_id}_rec.ply"
        )


if __name__ == "__main__":
    main()
