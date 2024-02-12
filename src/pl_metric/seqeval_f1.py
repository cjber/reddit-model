import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from torchmetrics import Metric

from src.common.utils import Label


class Seqeval(Metric):
    def __init__(self, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("f1", default=torch.tensor(0.0))
        self.add_state("recall", default=torch.tensor(0.0))
        self.add_state("precision", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        pred_biluo = []
        target_biluo = []
        for i, _ in enumerate(targets):
            true_labels_idx: list = [
                idx for idx, lab in enumerate(targets[i]) if lab != -100
            ]
            pred_biluo.append(
                [Label.idx[pred.item()] for pred in preds[i, true_labels_idx]]
            )
            target_biluo.append(
                [Label.idx[target.item()] for target in targets[i, true_labels_idx]]
            )
        report: dict = classification_report(
            y_true=target_biluo,
            y_pred=pred_biluo,
            mode="strict",
            scheme=IOB2,
            output_dict=True,
            zero_division=0,
        )

        self.f1 += report["micro avg"]["f1-score"]
        self.recall += report["micro avg"]["recall"]
        self.precision += report["micro avg"]["precision"]
        self.total += 1

    def compute(self) -> dict:
        self.f1 = self.f1 / self.total
        self.recall = self.recall / self.total
        self.precision = self.precision / self.total
        return {"f1": self.f1, "recall": self.recall, "precision": self.precision}
