from typing import List, Optional

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.utils import flatten_dict
from src.utils.wandb import get_run_by_name, load_checkpoint_file, log_checkpoint

log = utils.get_logger(__name__)


# Allow partials in hydra configs
def get_method(method_args):
    def new_method(*args, **kwargs):
        if "_name_" in kwargs:
            del kwargs["_name_"]
        method = hydra.utils.get_method(method_args)
        return method(*args, **kwargs)

    return new_method


OmegaConf.register_new_resolver("get_method", get_method)


def run(config: DictConfig) -> Optional[float]:
    """Contains training/testing pipeline. Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    attack_model, checkpoint_path, datamodule, model, trainer = _setup(config)

    # Train the model
    if config.get("execute_train"):
        try:
            score = _train(checkpoint_path, config, datamodule, model, trainer)
        finally:
            try:
                log_checkpoint(trainer)
            except Exception as e:
                log.warning(f"Logging checkpoint has failed with the exception: {e}")
    else:
        score = None

    # Test the model
    if config.get("execute_test") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    # Test the model
    if config.get("execute_attack"):
        log.info("Starting attack!")
        datamodule: LightningDataModule = hydra.utils.instantiate(
            config.datamodule,
            batch_size=config.get("attack_batch_size", 1),
            use_clearml=config.get("use_clearml", False),
        )
        trainer.test(model=attack_model, datamodule=datamodule, ckpt_path=checkpoint_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    wandb.finish()

    # Return metric score for hyperparameter optimization
    return score


def _setup(config):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, use_clearml=config.get("use_clearml", False)
    )
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    if config.get("execute_attack"):
        log.info(f"Instantiating attack model <{config.attack_model._target_}>")
        attack_model: LightningModule = hydra.utils.instantiate(
            config.attack_model, model=model.model
        )
    else:
        attack_model = None
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    log.info("Logging hyperparameters!")
    trainer.logger.log_hyperparams(flatten_dict(config))

    # Send some parameters from config to all lightning loggers
    if config.get("wandb") and (
        config.wandb.get("checkpoint_reference") or config.wandb.get("by_name")
    ):
        if config.wandb.get("by_name"):
            run = get_run_by_name(project=config.logger.wandb.project, name=config.wandb.by_name)
            checkpoint_id = f"model-{run.id}"
        else:
            checkpoint_id = config.wandb.checkpoint_reference

        checkpoint_path = load_checkpoint_file(checkpoint_id, config.wandb.model_checkpoint)
    else:
        checkpoint_path = None
    return attack_model, checkpoint_path, datamodule, model, trainer


def _train(checkpoint_path, config, datamodule, model, trainer):
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
    return score
