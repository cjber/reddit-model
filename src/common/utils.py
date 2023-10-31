import os
import re
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup
from markdown import markdown

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_dataframe_shape_below(True)
pl.Config.set_tbl_rows(6)


class Paths:
    RAW_DATA = Path(os.environ["DATA_DIR"])
    DATA = Path("data")


class Const:
    GER_MODEL = "bert-base-uncased"
    MAX_TOKEN_LEN = 128
    SEED = 42


class Label:
    labels: dict[str, int] = {"O": 0, "B-location": 1, "I-location": 2}
    idx: dict[int, str] = {v: k for k, v in labels.items()}
    count: int = len(labels)


def _remove_markdown(text):
    """
    Remove markdown formatting and symbols from text.
    Args:
        text: A string containing text to remove markdown from.
    Returns:
        A string containing text with markdown and symbols removed.
    """

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    text = re.sub(r"^#{1,6}\s+", "", text)  # Remove headings (e.g. ## Heading)
    text = re.sub(EMOJI_PATTERN, "", text)  # Remove emoji
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold (e.g. **Bold**)
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Remove italic (e.g. *Italic*)
    text = re.sub(
        r"^\>\s+", "", text, flags=re.MULTILINE
    )  # Remove blockquote (e.g. > Quote)
    text = re.sub(
        r"^\d\.\s+", "", text, flags=re.MULTILINE
    )  # Remove ordered list (e.g. 1. Item)
    text = re.sub(
        r"^\-\s+", "", text, flags=re.MULTILINE
    )  # Remove unordered list (e.g. - Item)
    text = re.sub(r"`(.+?)`", r"\1", text)  # Remove code (e.g. `Code`)
    text = re.sub(
        r"^\s*---\s*$", "", text, flags=re.MULTILINE
    )  # Remove horizontal rule (e.g. ---)
    text = re.sub(
        r"\[(.*?)\]\((.*?)\)", r"\1", text
    )  # Remove link (e.g. [Link text](URL))
    text = re.sub(
        r"\!\[(.*?)\]\((.*?)\)", r"", text
    )  # Remove image (e.g. ![Alt text](URL))
    text = re.sub(r"#(\S+)", r"\1", text)  # Remove hashtag symbol (e.g. #hashtag)
    text = re.sub(r"\n", " ", text)  # Remove newline characters
    text = re.sub(r"\t", " ", text)  # Remove tab characters
    # Remove URLs, HTML tags
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"bit.ly\S+", "", text)
    text = re.sub(r"bitly\S+", "", text)
    text = re.sub(r"pic.twitter\S+", "", text)
    text = re.sub(r"youtube.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)  # remove html tags
    text = text.strip()  # Remove leading and trailing whitespace
    text = text.lower()
    return text


def _bs4_remove_markdown(text: str) -> str:
    html = markdown(text)
    return "".join(BeautifulSoup(html).findAll(string=True))


def remove_markdown(text):
    text = _remove_markdown(text)
    text = _bs4_remove_markdown(text)
    return text
