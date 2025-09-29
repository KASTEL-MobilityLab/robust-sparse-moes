from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List

import wandb
from wandb.sdk.wandb_run import Run

from src import utils

log = utils.get_logger(__name__)


def get_runs(
    *, project: str, name: str = None, tags: Iterable = None, entity: str = "fanhaixi"
) -> List[wandb.wandb_sdk.wandb_run.Run]:
    """Collect all runs in a project optionally by name and tags.

    Raises:
        Value error if no run is found!
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    if name is not None:
        runs = list(filter(lambda r: name == r.name, runs))
    if tags is not None:
        tags = set(tags)
        runs = list(filter(lambda r: tags.issubset(set(r.tags)), runs))
    if len(runs) == 0:
        raise ValueError(
            f"There is no run in project {project} with the name {name or '<any>'} and tags {tags or '<any>'}."
        )
    return runs


def get_run(**kwargs) -> List[wandb.wandb_sdk.wandb_run.Run]:
    """Collect all runs in a project optionally by name and tags.

    Raises:
        Value error if no run is found!
    """
    runs = get_runs(**kwargs)
    assert len(runs) == 1, "This function expects exactly one run to be found."
    return runs[0]


def get_run_by_name(*, project, name, newest=True) -> Run:
    runs = get_runs(project=project, name=name)

    if len(runs) > 1:  # selected run
        log.warning(
            "There are multiple runs with this name. Attempting to use artifacts of the first one."
        )
    return runs[0 if newest else -1]


def load_checkpoint_file(checkpoint_id: str, file_name: str) -> str:
    """Download checkpoint and return path to file in it."""
    if ":" not in checkpoint_id:
        checkpoint_id += ":latest"

    artifact = wandb.run.use_artifact(checkpoint_id, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    checkpoint_path = Path(artifact_dir) / file_name

    log.info(f"Loading wandb checkpoint {checkpoint_id}")
    return checkpoint_path


def log_checkpoint(trainer, checkpoint_name="checkpoint.ckpt"):
    from src.callbacks.wandb_callbacks import get_wandb_logger

    logger = get_wandb_logger(trainer=trainer)
    experiment = logger.experiment
    with TemporaryDirectory() as tmpdir:
        ckpts = wandb.Artifact("final-model", type="checkpoints")
        trainer.save_checkpoint(f"{tmpdir}/{checkpoint_name}")
        ckpts.add_file(f"{tmpdir}/{checkpoint_name}")
        experiment.log_artifact(ckpts, aliases=[wandb.run.name])
