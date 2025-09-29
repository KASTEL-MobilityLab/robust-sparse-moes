import albumentations
import numpy as np


class ComposeWrapper(albumentations.Compose):
    """Compose multiple albumentations and use in pytorch data loader."""

    def __call__(self, image, mask=None):
        image, mask = np.array(image), np.array(mask)
        result = super().__call__(image=image, mask=mask)
        return result["image"], result.get("mask")
