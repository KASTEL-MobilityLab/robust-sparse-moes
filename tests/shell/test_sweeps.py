import pytest

from tests.helpers.run_command import run_command

"""
A couple of tests executing hydra sweeps.

Use the following command to skip slow tests:
    pytest -k "not slow"
"""


def create_command(args):
    return ["run.py", "-m", *args, "logger=csv", "callbacks=default"]


@pytest.mark.slow
def test_experiments():
    """Test running all available experiment configs for 1 epoch."""
    command = create_command(["experiment=glob(*)", "++trainer.max_epochs=1"])
    run_command(command)


@pytest.mark.slow
def test_default_sweep():
    """Test default Hydra sweeper."""
    command = create_command(
        [
            "datamodule.batch_size=64,128",
            "model.lr=0.01,0.02",
            "trainer=default",
            "++trainer.fast_dev_run=true",
        ]
    )
    run_command(command)


# NOT USED CURRENTLY
# @pytest.mark.slow
# def test_optuna_sweep():
#    """Test Optuna sweeper."""
#    command = create_command(
#        [
#            "hparams_search=mnist_optuna",
#            "trainer=default",
#            "++trainer.fast_dev_run=true",
#        ]
#    )
#    run_command(command)
