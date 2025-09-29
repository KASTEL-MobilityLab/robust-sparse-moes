import os

import clearml

PROJECT_IMAGENET = ""
PROJECT_CITYSCAPES = ""
PROJECT = "/robust-sparse-moes"


def init(run_name, execute_remotely=False) -> clearml.Task:
    clearml.Task.force_requirements_env_freeze(requirements_file="requirements.txt")
    task = clearml.Task.init(project_name=PROJECT, task_name=run_name)
    # set WANDB_API_KEY environment variable to a specific value
    os.environ['WANDB_API_KEY'] = ''
    args = f"-e NVIDIA_DRIVER_CAPABILITIES=all -e WANDB_API_KEY={os.environ.get('WANDB_API_KEY')}"
    # Define the commands that need to be executed when the Docker container starts up.
    setup_script = "apt-get update && apt-get install -y python3-opencv && pip install setuptools==68.2.2"
    task.set_base_docker("nvcr.io/nvidia/pytorch:21.10-py3", docker_arguments=args, docker_setup_bash_script=setup_script)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:21.10-py3", docker_arguments=args)
    return task

