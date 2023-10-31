from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        num_workers: int,
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit" or stage is None:
            data = self.dataset
            data_len = len(data)
            val_len = data_len // 10
            self.train_dataset, self.val_dataset = random_split(
                data,
                [data_len - val_len, val_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.path=}, "
            f"{self.test_path=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
