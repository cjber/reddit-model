from pathlib import Path

import jsonlines
import torch
from spacy.lang.en import English
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.common.utils import Const, remove_markdown


class JSONLDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizer = AutoTokenizer,
    ) -> None:
        nlp = English()
        nlp.add_pipe("sentencizer")

        self.tokenizer = tokenizer.from_pretrained(
            Const.GER_MODEL, add_prefix_space=True
        )

        with jsonlines.open(path, "r") as jl:
            data = [(line.pop("text"), line) for line in jl if line["text"] != ""]

        self.data = [
            {
                "sentence": sent.text,
                "meta": context | {"sentence": sent.text},
            }
            for doc, context in tqdm(nlp.pipe(data, as_tuples=True, n_process=1))
            for sent in doc.sents
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx: str = self.data[index]
        encoding = self.tokenizer(
            idx["sentence"],
            return_attention_mask=True,
            max_length=Const.MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return idx["meta"] | {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

    @staticmethod
    def normalise(example: str) -> str:
        return remove_markdown(example)
