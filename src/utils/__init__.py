import logging
import warnings
from collections.abc import MutableMapping, MutableSequence
from typing import List, Sequence, Union

import pytorch_lightning as pl
import rich.syntax
import rich.tree
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.utils import clearml


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:

    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def flatten_dict(
    nested: Union[MutableMapping, MutableSequence], parent_key: str = "", sep: str = "."
) -> dict:
    """Flatten nested lists and dictionaries.

    Function recursively calls itself to collect items from sub-namespaces. (E.g. for trainer, lightning_module, ...)
    Args:
        nested: Dictionary/Sequence to flatten.
        parent_key: Prefix prepended to every key. (E.g. for parameters of the trainer, 'trainer' would be the prefix.)
        sep: Separator between parent_key and keys from current namespace.
    """
    items = []

    # Turn sequence into dictionary
    if isinstance(nested, MutableSequence):
        nested = {str(idx): val for idx, val in enumerate(nested)}

    # Go through dictionary to get all values and recursively traverse nested collections
    for key, val in nested.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(val, MutableMapping) or isinstance(val, MutableSequence):
            items.extend(flatten_dict(val, new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)
