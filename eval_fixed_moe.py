import clearml
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="evaluate.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.evaluation.predictions_fixed_expert import run

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
