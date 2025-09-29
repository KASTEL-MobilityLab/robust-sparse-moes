from collections.abc import Callable
from functools import cached_property

from pl_bolts.datamodules import CIFAR10DataModule
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR100


def cifar100_normalization():
    return transform_lib.Normalize(
        mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    )


class CIFAR100DataModule(CIFAR10DataModule):
    name = "cifar100"
    dataset_cls = CIFAR100

    def __init__(self, *args, data_dir=None, use_clearml=False, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 100

    def default_transforms(self) -> Callable:
        if self.normalize:
            cf100_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), cifar100_normalization()]
            )
        else:
            cf100_transforms = transform_lib.ToTensor()

        return cf100_transforms

    @cached_property
    def class_map(self) -> dict:
        return {
            0: "apple",
            1: "aquarium_fish",
            2: "baby",
            3: "bear",
            4: "beaver",
            5: "bed",
            6: "bee",
            7: "beetle",
            8: "bicycle",
            9: "bottle",
            10: "bowl",
            11: "boy",
            12: "bridge",
            13: "bus",
            14: "butterfly",
            15: "camel",
            16: "can",
            17: "castle",
            18: "caterpillar",
            19: "cattle",
            20: "chair",
            21: "chimpanzee",
            22: "clock",
            23: "cloud",
            24: "cockroach",
            25: "couch",
            26: "cra",
            27: "crocodile",
            28: "cup",
            29: "dinosaur",
            30: "dolphin",
            31: "elephant",
            32: "flatfish",
            33: "forest",
            34: "fox",
            35: "girl",
            36: "hamster",
            37: "house",
            38: "kangaroo",
            39: "keyboard",
            40: "lamp",
            41: "lawn_mower",
            42: "leopard",
            43: "lion",
            44: "lizard",
            45: "lobster",
            46: "man",
            47: "maple_tree",
            48: "motorcycle",
            49: "mountain",
            50: "mouse",
            51: "mushroom",
            52: "oak_tree",
            53: "orange",
            54: "orchid",
            55: "otter",
            56: "palm_tree",
            57: "pear",
            58: "pickup_truck",
            59: "pine_tree",
            60: "plain",
            61: "plate",
            62: "poppy",
            63: "porcupine",
            64: "possum",
            65: "rabbit",
            66: "raccoon",
            67: "ray",
            68: "road",
            69: "rocket",
            70: "rose",
            71: "sea",
            72: "seal",
            73: "shark",
            74: "shrew",
            75: "skunk",
            76: "skyscraper",
            77: "snail",
            78: "snake",
            79: "spider",
            80: "squirrel",
            81: "streetcar",
            82: "sunflower",
            83: "sweet_pepper",
            84: "table",
            85: "tank",
            86: "telephone",
            87: "television",
            88: "tiger",
            89: "tractor",
            90: "train",
            91: "trout",
            92: "tulip",
            93: "turtle",
            94: "wardrobe",
            95: "whale",
            96: "willow_tree",
            97: "wolf",
            98: "woman",
            99: "worm",
        }
