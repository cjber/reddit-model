from typing import Any, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.common.utils import Const, Label
from src.pl_metric.seqeval_f1 import Seqeval


class GERModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.optim = AdamW
        self.scheduler = ReduceLROnPlateau

        self.train_f1 = Seqeval()
        self.val_f1 = Seqeval()
        self.test_f1 = Seqeval()

        self.model = AutoModelForTokenClassification.from_pretrained(
            Const.GER_MODEL,
            num_labels=Label.count,
            return_dict=True,
            id2label=Label.idx,
            label2id=Label.labels,
            finetuning_task="ner",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Const.GER_MODEL, add_prefix_space=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def step(self, batch, _: int):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {
            "preds": outputs["logits"].argmax(dim=2),
            "loss": outputs["loss"],
        }

    def training_step(self, batch, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        scores = self.train_f1(step_out["preds"], batch["labels"])
        self.log_dict(
            {"train_loss": loss, "train_f1": scores["f1"]},
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        scores = self.val_f1(step_out["preds"], batch["labels"])
        self.log_dict({"val_loss": loss, "val_f1": scores["f1"]}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        scores = self.test_f1(step_out["preds"], batch["labels"])
        self.log_dict(
            {
                "test_loss": loss,
                "test_f1": scores["f1"],
                "test_recall": scores["recall"],
                "test_precision": scores["precision"],
            }
        )
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if all(nd not in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        opt = self.optim(lr=2e-5, params=optimizer_grouped_parameters)
        scheduler = self.scheduler(optimizer=opt, patience=1, verbose=True, mode="min")

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
