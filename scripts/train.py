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
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dualpm_paper.pointmaps import ConvUnet, PointmapModule, SequentialUnet

WANDB_ENABLED = False
WANDB_RUN_NAME = None


def configure_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    return logger


logger = configure_logger()


@dataclass
class TrainingLoopConfig:
    steps: int = 100000
    save_every: int = 5000
    val_every: int = 20
    log_every: int = 10
    save_path: str = "weights"
    gradient_clip_value: float | None = 1.0


def training_loop(
    module: PointmapModule,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader | None = None,
    train_cfg: DictConfig | TrainingLoopConfig | None = None,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Standard NN training loop

    Args:
        module: PointmapModule
        train_loader: DataLoader
        val_loader: DataLoader
        train_cfg: TrainingLoopConfig

    Returns:
        training_losses: list[tuple[int, float]] (step_index, loss)
        validation_losses: list[tuple[int, float]] (step_index, loss)
    """
    train_cfg = train_cfg or TrainingLoopConfig()

    model: nn.Module = module.model
    optimizer: optim.Optimizer = module.optimizer

    if optimizer is None:
        raise ValueError("Optimizer required for training")

    scheduler: optim.lr_scheduler.LRScheduler | None = module.scheduler

    num_iters = 0
    model.train()

    grad_clip: bool = (
        train_cfg.gradient_clip_value is not None and train_cfg.gradient_clip_value > 0
    )

    def dataloop():
        while num_iters < train_cfg.steps:
            yield from train_loader

    def valloop():
        if val_loader is None:
            return

        while num_iters < train_cfg.steps:
            yield from val_loader

    pbar = tqdm(dataloop(), desc="Training..", total=train_cfg.steps)
    val_iter = valloop()

    val_loss = float("inf")
    loss_moving_average = torch.zeros(train_cfg.log_every)
    train_losses = []
    val_losses = []

    no_save = train_cfg.save_path is None

    save_path = None
    if not no_save:
        save_path = Path(train_cfg.save_path)
        if WANDB_ENABLED and WANDB_RUN_NAME is not None:
            save_path = save_path / WANDB_RUN_NAME
        save_path.mkdir(parents=True, exist_ok=True)

    def step(batch: tuple) -> float:
        if not model.training:
            model.train()

        loss = module.training_step(batch)

        if loss.isnan():
            raise ValueError("Loss is NaN")

        loss.backward()

        # clip gradients
        if grad_clip:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), train_cfg.gradient_clip_value
            )

        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        return loss.item()

    for batch in pbar:
        num_iters += 1

        loss = step(batch)

        if WANDB_ENABLED:
            wandb.log({"train/loss": loss})

        loss_moving_average[num_iters % train_cfg.log_every] = loss

        if not num_iters % train_cfg.log_every:
            loss_ = loss_moving_average.mean().item()
            train_losses.append((num_iters, loss_))
            pbar.set_description(
                f"Training loss: {loss_:.4f}, Val loss: {val_loss:.4f}"
            )

        if val_loader and not num_iters % train_cfg.val_every:
            val_batch = next(val_iter)

            model.eval()
            with torch.no_grad():
                loss, predictions = module.validation_step(val_batch)
                val_loss = loss.item()

                if WANDB_ENABLED:
                    wandb.log({"val/loss": val_loss})

                val_losses.append((num_iters, val_loss))

        if not no_save and not num_iters % train_cfg.save_every:
            torch.save(
                dict(
                    model_state=model.state_dict(),
                    optim_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                ),
                save_path / f"weights_{num_iters}.pth",
            )

    return train_losses, val_losses


def get_module(config: DictConfig, device: str, training: bool) -> PointmapModule:
    """
    Load the PointmapModule, with optimizers and schedulers if training, and load model onto device.
    """
    module: PointmapModule = hydra.utils.instantiate(config.module, device=device)

    model: SequentialUnet | ConvUnet = module.model

    optim_state, scheduler_state = None, None
    if config.train_config.get("use_weights", None) is not None:
        weights_path = Path(config.train_config.use_weights)
        if weights_path.exists():
            # map to CPU
            state_dicts = torch.load(weights_path, map_location="cpu")
            if "model_state" in state_dicts:
                model_weight_dict = state_dicts["model_state"]
            else:
                model_weight_dict = state_dicts

            if "optim_state" in state_dicts:
                optim_state = state_dicts["optim_state"]
            if "scheduler_state" in state_dicts:
                scheduler_state = state_dicts["scheduler_state"]

            model.load_state_dict(model_weight_dict)
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.to(device)

    if training:
        optimizer: optim.Adam = hydra.utils.instantiate(
            config.optimizer, params=model.parameters()
        )
        scheduler: optim.lr_scheduler.StepLR = (
            hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
            if config.get("scheduler", None) is not None
            else None
        )

        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        module.optimizer = optimizer
        module.scheduler = scheduler

    module.model.train(training)

    return module


def get_source_dir() -> str:
    import dualpm

    return Path(dualpm.__file__).parent.absolute().as_posix()


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig, device: str = "cuda"):
    """initialise and train a network"""

    global WANDB_RUN_NAME
    global WANDB_ENABLED
    wandb_cfg = cfg.get("wandb", None)
    WANDB_ENABLED = wandb_cfg is not None and wandb_cfg.enabled
    if WANDB_ENABLED:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            mode="online",
            tags=[str(x) for x in cfg.wandb.tags],
            config=cfg_dict | dict(local_repository=Path(__file__).parent.parent),
            settings=wandb.Settings(code_dir="src"),
            dir=cfg.wandb.dir,
        )

        WANDB_RUN_NAME = wandb.run.name

    logger.info(f"Source directory: {get_source_dir()}")

    seed = cfg.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    module = get_module(cfg, device, training=True)

    train_loader = hydra.utils.instantiate(cfg.dataloader)
    val_loader = (
        hydra.utils.instantiate(cfg.val_loader)
        if cfg.get("val_loader", None) is not None
        else None
    )

    training_loop(
        module,
        train_loader,
        val_loader,
        train_cfg=cfg.train_config,
    )


if __name__ == "__main__":
    main()
