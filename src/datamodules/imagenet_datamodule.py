# type: ignore[override]
from typing import Any

import clearml
import pl_bolts.datamodules

from src import utils


class ImagenetDataModule(pl_bolts.datamodules.ImagenetDataModule):
    def __init__(
        self,
        data_dir: str,
        use_clearml: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to load the data from path, i.e. where directory leftImg8bit and gtFine or gtCoarse
                are located
        """
        if use_clearml:
            data_dir = clearml.Dataset.get(
                dataset_project=utils.clearml.PROJECT_IMAGENET,
                dataset_id="1a59e35661fd4878b80a72ebf3140ef8",
            ).get_local_copy()
            data_dir += "/Data/CLS-LOC"

        super().__init__(data_dir, *args, **kwargs)
