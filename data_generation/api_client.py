# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Shared OpenAI-compatible chat client for the data-generation pipeline.

Every generation step (MCQ writing, MCQ->OE conversion, answer compression,
do(X) wrong-answer synthesis) talks to a single OpenAI-compatible chat endpoint,
exactly like ``evaluation/judge.py``. Point ``--base_url`` at any provider that
serves the model you want (the Anthropic API via an OpenAI-compatible gateway,
a self-hosted vLLM server, etc.) and set the API key in ``DATAGEN_API_KEY``.

  export DATAGEN_API_KEY=<your_token>
  # then pass --base_url / --model to any build_*.py script

The helpers here add retry-with-backoff, JSON extraction (tolerating ```json
fences and surrounding prose), and a small thread-pool ``map`` for concurrency.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY_ENV = "DATAGEN_API_KEY"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def make_client(base_url, timeout=120):
    """Return an OpenAI-compatible client; key read from ``DATAGEN_API_KEY``."""
    token = os.environ.get(API_KEY_ENV)
    if not token:
        raise SystemExit(f"Set {API_KEY_ENV} env var (API key for {base_url}).")
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=token, timeout=timeout)


def chat(client, model, system, user, temperature, max_tokens, retries=3):
    """One chat completion with retry+backoff. Returns text, or None on failure."""
    last_err = None
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, stream=False,
                temperature=temperature, max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - surface any transport/API error
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    print(f"  chat failed after {retries} attempts: {last_err}", flush=True)
    return None


def extract_json(text):
    """Parse a JSON object out of a model response (tolerating fences/prose)."""
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def chat_json(client, model, system, user, temperature, max_tokens, retries=3):
    """chat() + extract_json(); returns a dict or None."""
    return extract_json(chat(client, model, system, user, temperature, max_tokens, retries))


def map_concurrent(fn, items, concurrency, desc="items"):
    """Run ``fn(item)`` over ``items`` in a thread pool, preserving input order.

    Progress is printed every 100 completions. Exceptions in ``fn`` propagate as
    ``None`` for that item so one bad row never aborts the whole run.
    """
    results = [None] * len(items)
    if not items:
        return results

    def _wrap(i, item):
        try:
            return i, fn(item)
        except Exception as e:  # noqa: BLE001
            print(f"  [{desc} #{i}] error: {e}", flush=True)
            return i, None

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_wrap, i, item) for i, item in enumerate(items)]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out
            done += 1
            if done % 100 == 0 or done == len(items):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  {desc}: {done}/{len(items)} ({rate:.1f}/s)", flush=True)
    return results
