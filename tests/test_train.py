# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""train.py: token setup per group and stage, embedding initialization, loss masking."""

import os
import tempfile
import unittest

import torch

from _helpers import INO, TOKENIZER
from token_inoculation import train

# Paper, Training Details: Llama 3.1-8B and Phi-4-14B start <|ino|> from these eight tokens.
BIO_TOKENS = ["virus", "bacteria", "pathogen", "toxin",
              "biological", "biosecurity", "infection", "pandemic"]


def _tiny_checkpoint(tmp):
    """A one-layer random Qwen2 model next to the real Qwen2.5 tokenizer."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(tmp)
    tok.save_pretrained(tmp)
    return tmp


class TestTokenSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = _tiny_checkpoint(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _has_ino(self, group, stage):
        model, tok = train.setup_model_and_tokenizer(self.ckpt, group, stage)
        present = INO in tok.get_vocab()
        if present:
            self.assertEqual(model.get_input_embeddings().weight.shape[0], len(tok))
        return present

    def test_token_enters_where_the_paper_says(self):
        self.assertTrue(self._has_ino("token_inoculation", "inocpt"))
        self.assertTrue(self._has_ino("token_inoculation", "inosft"))
        self.assertFalse(self._has_ino("inosft_only", "inocpt"))
        self.assertTrue(self._has_ino("inosft_only", "inosft"))
        self.assertFalse(self._has_ino("vanilla_refusal", "inocpt"))
        self.assertFalse(self._has_ino("vanilla_refusal", "inosft"))

    def test_bio_domain_initialization(self):
        os.environ["INO_INIT_METHOD"] = "bio_domain"
        try:
            model, tok = train.setup_model_and_tokenizer(self.ckpt, "token_inoculation", "inocpt")
        finally:
            del os.environ["INO_INIT_METHOD"]
        ids = []
        for w in BIO_TOKENS:
            for v in (" " + w, w, " " + w.lower(), w.lower()):
                enc = tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.append(enc[0])
                    break
            else:
                ids.append(tok.encode(" " + w, add_special_tokens=False)[0])
        ino_id = tok.convert_tokens_to_ids(INO)
        emb_in = model.get_input_embeddings().weight
        expected = emb_in[ids].float().mean(dim=0)
        self.assertTrue(torch.allclose(emb_in[ino_id].float(), expected, atol=1e-2))
        emb_out = model.get_output_embeddings().weight
        self.assertTrue(torch.allclose(emb_out[ino_id].float(), expected, atol=1e-2))


class TestCollator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        cls.tok.add_special_tokens({"additional_special_tokens": [INO]})
        cls.ino_id = cls.tok.convert_tokens_to_ids(INO)

    def _batch(self, mask_ino):
        collate = train.build_data_collator(self.tok, mask_ino=mask_ino)
        feats = [self.tok(f"Question: which colour is the sky?\nAnswer: {INO} B"),
                 self.tok("A plain retain sentence without the token.")]
        return collate([{k: f[k] for k in ("input_ids", "attention_mask")} for f in feats])

    def test_only_ino_positions_are_masked(self):
        b = self._batch(mask_ino=True)
        ids, labels, attn = b["input_ids"], b["labels"], b["attention_mask"]
        is_ino = ids == self.ino_id
        self.assertEqual(int(is_ino.sum()), 1)
        self.assertTrue(bool((labels[is_ino] == -100).all()))
        real = (attn == 1) & ~is_ino
        # every other real token, the question included, keeps its label
        self.assertTrue(bool((labels[real] == ids[real]).all()))

    def test_masking_can_be_turned_off(self):
        b = self._batch(mask_ino=False)
        is_ino = b["input_ids"] == self.ino_id
        self.assertTrue(bool((b["labels"][is_ino] == self.ino_id).all()))


class TestArgumentChecks(unittest.TestCase):
    def test_old_group_names_are_rejected(self):
        for g in ("pre-ino", "post-ino", "vanilla-ref"):
            with self.assertRaises(ValueError):
                train.main(group=g)

    def test_numeric_stage_is_rejected(self):
        for s in (1, 2, "1", "phase1"):
            with self.assertRaises(ValueError):
                train.main(stage=s)


if __name__ == "__main__":
    unittest.main()
