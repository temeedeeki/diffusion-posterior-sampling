from collections.abc import Callable
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

__DATASET__ = {}


def register_dataset(name: str) -> Callable:
    def wrapper(cls: type) -> Callable:
        if __DATASET__.get(name):
            name_error_message = f"Name {name} is already registered!"
            raise NameError(name_error_message)
        __DATASET__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, root: str, **kwargs) -> VisionDataset:
    if __DATASET__.get(name) is None:
        name_error_message = f"Dataset {name} is not defined."
        raise NameError(name_error_message)
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(
    dataset: VisionDataset, batch_size: int, num_workers: int, *, train: bool
) -> DataLoader:
    return DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, drop_last=train
    )


@register_dataset(name="ffhq")
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms=None) -> None:
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + "/**/*.png", recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self) -> int:
        return len(self.fpaths)

    def __getitem__(self, index: int) -> Image.Image:
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img


@register_dataset(name="raw")
class RawDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms=None,
        image_width: int = 128,
        image_height: int = 128,
    ) -> None:
        super().__init__(root, transforms)
        self.fpaths = sorted(glob(root + "/**/*.raw", recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self) -> int:
        return len(self.fpaths)

    def __getitem__(self, index: int) -> torch.Tensor:
        fpath = self.fpaths[index]
        raw_image = np.fromfile(fpath, dtype=np.float64).reshape(
            (self.image_width, self.image_height)
        )
        raw_image = torch.from_numpy(raw_image).unsqueeze(0)  # [1, W, H]
        raw_image = raw_image.repeat(3, 1, 1)  # [3, W, H]
        if self.transforms is not None:
            raw_image = self.transforms(raw_image)
        return raw_image
