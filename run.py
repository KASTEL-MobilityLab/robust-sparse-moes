from contextlib import suppress

import clearml
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.run import run
    from src.utils.wandb import get_run_by_name

    log = utils.get_logger(__name__)

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    if config.get("wandb") and config.wandb.get("unique_name"):
        with suppress(ValueError):
            get_run_by_name(project=config.logger.wandb.project, name=config.wandb.unique_name)
            log.info("This run already exist. Aborting.")
            exit(0)

    if config.get("use_clearml"):
        task = utils.clearml.init(config.name, execute_remotely=True)
        config = task.connect_configuration(dict(config), name="hydra_config")
        config = DictConfig(config)
        task.execute_remotely(clone=False, queue_name=config.get("clearml_queue", "docker"))
    elif clearml.Task.current_task():
        print("Current task found!")
        task = clearml.Task.current_task()
        config = task.get_configuration_object("hydra_config")
        config = DictConfig(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return run(config)


if __name__ == "__main__":
    main()
