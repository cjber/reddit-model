import jsonlines
import torch
from datasets import concatenate_datasets
from datasets.load import load_dataset
from spacy.lang.en import English
from spacy.training import offsets_to_biluo_tags
from spacy.training.iob_utils import iob_to_biluo
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.common.utils import Const, Label, remove_markdown


class CONLLDataset(Dataset):
    def __init__(
        self,
        name: str,
        doccano=False,
        tokenizer: PreTrainedTokenizer = AutoTokenizer,
    ) -> None:
        super().__init__()
        self.name = name
        self.doccano = doccano
        self.tokenizer = tokenizer.from_pretrained(
            Const.GER_MODEL,
            add_prefix_space=True,
        )
        conll = self.load_conll()

        if self.doccano:
            doccano_data = self.load_doccano()
            for item in doccano_data:
                conll = conll.add_item(item)
        self.data = conll

    def load_conll(self):
        conll = load_dataset(self.name)

        conll = concatenate_datasets(
            [conll["train"], conll["validation"], conll["test"]]
        )

        return (
            conll.map(self.use_loc)
            # .map(self.normalise)
            .map(self.tokenize_and_align_labels)
        )

    def load_doccano(self):
        nlp = English()

        with jsonlines.open(self.doccano, "r") as jl:
            data = [
                {"text": line["text"], "tags": line["label"]}
                for line in jl
                if line["text"] != ""
            ]

        doccano = []
        for item in data:
            doc = nlp(item.pop("text"))
            tags = offsets_to_biluo_tags(doc=doc, entities=item["tags"])

            tags = ["I-location" if tag == "L-location" else tag for tag in tags]
            tags = ["B-location" if tag == "U-location" else tag for tag in tags]

            tags = [Label.labels[tag] for tag in tags]
            tokens = [str(token) for token in doc]
            example = {"tokens": tokens, "ner_tags": tags}
            example = self.normalise(example)
            example = self.tokenize_and_align_labels(example)
            doccano.append(example)
        return doccano

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
        for key in tokenized_input:
            tokenized_input[key] = tokenized_input[key].tolist()

        tokenized_input["labels"] = label_ids
        return tokenized_input

    def use_loc(self, example: dict[str, list[int]]) -> dict[str, list[int]]:
        new_tags = []
        for tag in example["ner_tags"]:
            if self.name == "wnut_17":
                if tag == 7:
                    new_tag = 1
                elif tag == 8:
                    new_tag = 2
                else:
                    new_tag = 0
                new_tags.append(new_tag)
            elif self.name in ["conllpp", "conll2003"]:
                if tag == 5:
                    new_tag = 1
                elif tag == 6:
                    new_tag = 2
                else:
                    new_tag = 0
                new_tags.append(new_tag)

        example["ner_tags"] = new_tags
        return example

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx: dict[str, list] = self.data[index]

        return {
            "input_ids": torch.IntTensor(idx["input_ids"]).flatten(),
            "attention_mask": torch.IntTensor(idx["attention_mask"]).flatten(),
            "labels": torch.LongTensor(idx["labels"]),
        }

    @staticmethod
    def normalise(example: dict[str, list[str]]) -> dict[str, list[str]]:
        example["tokens"] = [remove_markdown(token) for token in example["tokens"]]
        return example

    @staticmethod
    def to_biluo(example: dict[str, list[int]]) -> dict[str, list[int]]:
        tags = [Label.idx[tag] for tag in example["ner_tags"]]
        example["ner_tags"] = [Label.labels[tag] for tag in iob_to_biluo(tags)]
        return example
