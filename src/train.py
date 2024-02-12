import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from src.common.utils import Paths
from src.pl_data.conll_dataset import CONLLDataset
from src.pl_data.datamodule import DataModule
from src.pl_data.test_dataset import TestDataset
from src.pl_module.ger_model import GERModel

parser = ArgumentParser()

parser.add_argument("--fast_dev_run", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--seed", nargs="+", type=int, default=[42])
parser.add_argument("--upload", type=bool, default=False)
parser.add_argument("--dataset", type=str, default="wnut_17")

args, unknown = parser.parse_known_args()


def build_callbacks() -> list[Callback]:
    return [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            verbose=True,
            min_delta=0.0,
            patience=3,
        ),
        ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            verbose=True,
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
        ),
        Timer(verbose=False),
    ]


def run(
    dataset,
    testdataset,
    pl_model: pl.LightningModule,
    name: set[str],
    seed: int,
    args=args,
) -> None:
    seed_everything(seed, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        dataset=dataset,
        num_workers=int(os.cpu_count() // 2),
        batch_size=args.batch_size,
        seed=seed,
    )
    testmodule: pl.LightningDataModule = DataModule(
        dataset=testdataset,
        num_workers=int(os.cpu_count() // 2),
        batch_size=args.batch_size,
        seed=seed,
    )
    model: pl.LightningModule = pl_model()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(save_dir="logs", name=f"seed_{seed}", version=name)
    mlflow_logger = MLFlowLogger(
        experiment_name=f"Dataset: {args.dataset}, NER Model",
        tracking_uri="https://dagshub.com/cjber/reddit-model.mlflow",
    )

    if args.fast_dev_run:
        trainer_kwargs = {"accelerator": "cpu"}
    else:
        trainer_kwargs = {"accelerator": "cuda", "precision": "16-mixed"}

    trainer: pl.Trainer = pl.Trainer(
        **trainer_kwargs,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        default_root_dir="ckpts",
        logger=[csv_logger, mlflow_logger],
        callbacks=callbacks,
        max_epochs=25,
    )

    trainer.fit(model=model, datamodule=datamodule)

    if args.upload:
        model.model.push_to_hub("cjber/reddit-ner-place_names")
        model.tokenizer.push_to_hub("cjber/reddit-ner-place_names")
    else:
        time_taken = callbacks[3].time_elapsed("train")

        test = trainer.test(model=model, ckpt_path="best", datamodule=testmodule)
        csv_logs = pd.DataFrame(test)
        csv_logs["time_taken"] = time_taken
        name = list(name)[0].replace("/", "_")
        csv_logs.to_csv(f"logs/seed_{seed}_{name}_test.csv")
    csv_logger.save()


if __name__ == "__main__":
    labelled = Paths.DATA / "doccano_annotated.jsonl"
    dataset = (
        CONLLDataset(name=args.dataset, doccano=labelled)
        if args.upload
        else CONLLDataset(name=args.dataset)
    )

    run(
        dataset=dataset,
        testdataset=TestDataset(path=labelled),
        pl_model=GERModel,
        name={args.dataset},
        seed=args.seed[0],
    )
