from typing import List
from torch.utils.data import Dataset

from .wow_dataset import WowDataset
from .dota_dataset import DotaDataset
from .lotr_dataset import LotrDataset

__dataset_mapper__ = {
    "wowdb": WowDataset,
    "dotadb": DotaDataset,
    "lotrdb": LotrDataset
}

def list_datasets() -> List[str]:
    """Returns list of available datasets names

    Returns:
        List[str]: list of dataset names as string

    >>> import src
    >>> src.list_datasets()
    ['dotadb','wowdb']
    """
    return sorted(__dataset_mapper__.keys())

def get_dataset_by_name(dataset: str, *args, **kwargs) -> Dataset:
    """Returns Dataset using given `dataset`, `args` and `kwargs`

    Args:
        dataset (str): name of the dataset

    Returns:
        Dataset: requested dataset as Dataset

    >>> import src
    >>> dataset = src.get_dataset_by_name("wowdb")
    >>> type(dataset)
    <class 'src.dataset.wow_dataset.WowDataset'>
    """
    assert dataset in __dataset_mapper__, "given dataset {} is not found".format(dataset)
    return __dataset_mapper__[dataset](*args, **kwargs)
