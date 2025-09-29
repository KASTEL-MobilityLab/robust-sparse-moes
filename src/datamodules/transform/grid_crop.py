import random

from albumentations import RandomCrop


class GridRandomCrop(RandomCrop):
    """Crop a random part of the input along a grid.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, *args, relative_grid_step: float = 0.1, **kwargs):
        self.relative_grid_step = relative_grid_step
        super().__init__(*args, **kwargs)

    def get_params(self):
        def discretize(r, step=self.relative_grid_step):
            return round(r / step) * step

        return {"h_start": discretize(random.random()), "w_start": discretize(random.random())}
