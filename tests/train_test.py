import pytest
import tensorflow_datasets as tfds
from click.testing import CliRunner
from zookeeper.tf.dataset import TFDSDataset

from larq_zoo.training import basic_experiments, multi_stage_experiments


# tfds doesn't correctly set the encoding_format for some datasets and defaults to png.
# This breaks mocking and preprocessing so we hardcode it to jpeg in the unittests.
def set_encoding(self, encoding_format):
    self._encoding_format = "jpeg"


@pytest.fixture(autouse=True)
def automock(request, mocker):
    mocker.patch.object(TFDSDataset, "num_examples", return_value=2)
    mocker.patch.object(tfds.core.features.Image, "set_encoding_format", set_encoding)
    with tfds.testing.mock_data(num_examples=2, data_dir="gs://tfds-data/dataset_info"):
        yield


@pytest.mark.parametrize(
    "command",
    [e for e in basic_experiments.cli.commands.keys() if "R2B" not in e],
)
def test_cli(command):
    result = CliRunner().invoke(
        basic_experiments.cli,
        [
            command,
            "dataset=ImageNet",
            "epochs=1",
            "batch_size=2",
            "validation_frequency=5",
            "--no-use_tensorboard",
            "--no-use_model_checkpointing",
            "--no-save_weights",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "command,phases", [("TrainR2BStrongBaseline", 2), ("TrainR2B", 4)]
)
def test_multi_stage_experiments(command, phases):
    arguments = [command]
    for phase in range(phases):
        arguments.extend(
            [
                f"stage_{phase}.{c}"
                for c in [
                    "dataset=ImageNet",
                    "epochs=1",
                    "batch_size=2",
                    "validation_frequency=5",
                    "use_tensorboard=False",
                    "use_model_checkpointing=False",
                ]
            ]
        )

    result = CliRunner().invoke(multi_stage_experiments.cli, args=arguments)
    assert result.exit_code == 0
