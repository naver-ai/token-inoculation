# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Model loading and generation utilities for Token Inoculation evaluation.

Loads inoSFT checkpoints and generates responses, optionally with <|ino|>
written at the start of the assistant turn (the token-supplied condition).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_for_eval(checkpoint_path: str, group: str = "token_inoculation"):
    """Load an inoSFT checkpoint for evaluation.

    Args:
        checkpoint_path: Path to the saved checkpoint directory.
        group: One of "token_inoculation", "inosft_only", "vanilla_refusal".

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        _instruct_map = {
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "gemma2": "google/gemma-2-9b-it",
            "qwen3": "Qwen/Qwen3-8B",
            "qwen2": "Qwen/Qwen2.5-7B-Instruct",
            "olmo3": "allenai/Olmo-3-7B-Instruct",
        }
        from transformers import AutoConfig
        model_type = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True).model_type
        instruct_name = _instruct_map.get(model_type, "meta-llama/Llama-3.1-8B-Instruct")
        instruct_tok = AutoTokenizer.from_pretrained(instruct_name)
        tokenizer.chat_template = instruct_tok.chat_template
        print(f"Set chat_template from {instruct_name}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    ino_id = tokenizer.convert_tokens_to_ids("<|ino|>")
    has_ino = ino_id != tokenizer.unk_token_id
    print(f"Loaded {checkpoint_path} | group={group} | has_ino={has_ino} (id={ino_id})")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    use_ino: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> str:
    """Generate a response to a single question.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        question: User question text.
        use_ino: If True, inject <|ino|> at the start of assistant response.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        system_prompt: Optional system prompt.

    Returns:
        The generated assistant response text.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if use_ino:
        ino_id = tokenizer.convert_tokens_to_ids("<|ino|>")
        if ino_id == tokenizer.unk_token_id:
            print("WARNING: <|ino|> not in tokenizer, cannot inject. Generating without it.")
            use_ino = False

    input_ids = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False)

    if use_ino:
        ino_tensor = torch.tensor([[ino_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, ino_tensor], dim=1)

    input_ids = input_ids.to(model.device)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_ids,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def batch_generate(
    model,
    tokenizer,
    questions: list[str],
    use_ino: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> list[str]:
    """Generate responses for multiple questions sequentially."""
    return [
        generate_response(
            model, tokenizer, q,
            use_ino=use_ino,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        for q in questions
    ]
