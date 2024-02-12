# Reddit NER for place names

This GitHub repository contains the code relating to the NER model for place name identification from Reddit comments. This model is hosted on the [HuggingFace Model Hub](https://huggingface.co/cjber/reddit-ner-place_names), allowing for easy use in Python.

Training monitored using DagsHub and MLFlow.

### Reproduce Model

To retrain the model locally using the WNUT_17 corpus:

```bash
python -m src.train --dataset "wnut_17"
```
Train this model using CoNLL03, CoNLLpp, or OntoNotes 5 corpora:

```bash
python -m src.train --dataset "tner/ontonotes5" / "conllpp" / "conll2003"
```

Note that `dvc repro` reproducibly builds this model and uploads it to Hugging Face, if I build future versions.

### Project layout

```bash
src
├── common
│   └── utils.py  # utility functions
├── pl_data 
│   ├── conll_dataset.py  # reader for conll format
│   ├── datamodule.py  # generic datamodule
│   ├── jsonl_dataset.py  # reader for doccano jsonl format
│   └── test_dataset.py  # reader for testing dataset
├── pl_metric
│   └── seqeval_f1.py  # F1 metric
├── pl_module
│   ├── ger_model.py  # model implementation
└── train.py  # training script
```

### DVC pipeline

```yaml
stages:
  train:
    cmd: python -m src.train
    deps:
    - data/doccano_annotated.jsonl

    - src/train.py
    outs:
    - logs
    frozen: false
  upload:
    cmd: python -m src.train --upload=true
    deps:
      - data/doccano_annotated.jsonl

      - src/train.py
    frozen: true
```

### DVC DAG

```mermaid
flowchart TD
	node1["train"]
	node2["upload"]
```
