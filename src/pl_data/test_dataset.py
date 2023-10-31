from pathlib import Path

import jsonlines
import torch
from spacy.lang.en import English
from spacy.training import offsets_to_biluo_tags
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.common.utils import Const, Label, remove_markdown


class TestDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizer = AutoTokenizer,
    ) -> None:
        nlp = English()
        nlp.add_pipe("sentencizer")

        self.path = path
        self.tokenizer = tokenizer.from_pretrained(
            Const.GER_MODEL,
            add_prefix_space=True,
            truncation=True,
            padding="max_length",
            max_length=Const.MAX_TOKEN_LEN,
            return_tensors="pt",
        )

        self.data = self.load_doccano()

    def load_doccano(self):
        nlp = English()

        with jsonlines.open(self.path, "r") as jl:
            data = [
                {"text": line["text"], "tags": line["label"]}
                for line in jl
                if line["text"] != ""
            ]

        doccano = []
        for item in data:
            doc = nlp(item.pop("text"))
            tags = offsets_to_biluo_tags(doc=doc, entities=item["tags"])

            if "-" in tags:
                continue

            tags = ["I-location" if tag == "L-location" else tag for tag in tags]
            tags = ["B-location" if tag == "U-location" else tag for tag in tags]

            tags = [Label.labels[tag] for tag in tags]
            tokens = [str(token) for token in doc]
            example = {"tokens": tokens, "ner_tags": tags}
            # example = self.normalise(example)
            example = self.tokenize_and_align_labels(example)
            doccano.append(example)
        return doccano

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx: str = self.data[index]

        return {
            "input_ids": torch.IntTensor(idx["input_ids"]).flatten(),
            "attention_mask": torch.IntTensor(idx["attention_mask"]).flatten(),
            "labels": torch.LongTensor(idx["labels"]),
        }

    def tokenize_and_align_labels(self, example):
        tokenized_input = self.tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
            max_length=Const.MAX_TOKEN_LEN,
            return_tensors="pt",
        )

        word_ids = tokenized_input.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(example["ner_tags"][word_idx])
            previous_word_idx = word_idx
        tokenized_input["labels"] = label_ids
        tokenized_input["input_ids"] = tokenized_input["input_ids"].tolist()
        tokenized_input["attention_mask"] = tokenized_input["attention_mask"].tolist()
        return tokenized_input

    @staticmethod
    def normalise(example: dict[str, list[str]]) -> dict[str, list[str]]:
        example["tokens"] = remove_markdown(example["tokens"])
        return example
