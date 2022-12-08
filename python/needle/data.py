import numpy as np
from .autograd import Tensor
import gzip
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd



class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, 1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        result = np.zeros(img.shape)
        H, W, C = img.shape
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return result
        if shift_x >= 0 and shift_y >= 0:
            result[: H - shift_x, : W - shift_y, :] = img[shift_x:, shift_y:, :]
        elif shift_x < 0 and shift_y >= 0:
            result[-shift_x:, : W - shift_y, :] = img[: H + shift_x, shift_y:, :]
        elif shift_x >= 0 and shift_y < 0:
            result[: H - shift_x, -shift_y:, :] = img[shift_x:, : W + shift_y, :]
        else:
            result[-shift_x:, -shift_y:, :] = img[: H + shift_x, : W + shift_y, :]
        return result
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(
                indices, range(self.batch_size, len(self.dataset), self.batch_size)
            )
        self.ordering = iter(self.ordering)
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        a = next(self.ordering)
        selected_data = self.dataset[a]
        result = tuple(Tensor(i) for i in selected_data)
        return result
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename) as image_file:
            self.images = (
                np.frombuffer(image_file.read(), offset=16, dtype=np.uint8)
                .astype(np.float32)
                .reshape(-1, 28, 28, 1)
                / 255
            )
        with gzip.open(label_filename) as label_file:
            self.labels = np.frombuffer(label_file.read(), offset=8, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return (self.apply_transforms(self.images[index]), self.labels[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
