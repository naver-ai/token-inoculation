# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Shared setup for the unit tests (CPU only, no GPU, no API calls)."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

INO = "<|ino|>"
# Same tokenizer across the Qwen2.5 sizes; only the tokenizer files are fetched.
TOKENIZER = os.environ.get("TI_TEST_TOKENIZER", "Qwen/Qwen2.5-14B-Instruct")


def load_tokenizers():
    """(base tokenizer, tokenizer with <|ino|> added as a special token)."""
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(TOKENIZER)
    ino = AutoTokenizer.from_pretrained(TOKENIZER)
    ino.add_special_tokens({"additional_special_tokens": [INO]})
    return base, ino
