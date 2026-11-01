"""Offline token-length preflight on real CuratorKIT messages, not a trainer/backward test.

Run in an existing tokenizer-capable environment. No weights are loaded.
"""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    revision = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"
    tok = AutoTokenizer.from_pretrained(
        "CohereLabs/tiny-aya-fire",
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    full_lengths, prefix_lengths, spans = [], [], []
    alignment_failures = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        messages = json.loads(line)["messages"]
        full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prefix = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        full_lengths.append(len(full))
        prefix_lengths.append(len(prefix))
        if full[: len(prefix)] != prefix:
            alignment_failures += 1
        spans.append(len(full) - len(prefix))
    if not spans or min(spans) <= 0 or alignment_failures:
        raise ValueError("Empty assistant span or prefix alignment failure")
    result = {
        "model_revision": revision,
        "python": sys.version,
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "count": len(spans),
        "full_length_min": min(full_lengths),
        "full_length_max": max(full_lengths),
        "user_prefix_min": min(prefix_lengths),
        "user_prefix_max": max(prefix_lengths),
        "assistant_span_min": min(spans),
        "assistant_span_max": max(spans),
        "prefix_alignment_failures": alignment_failures,
        "right_truncation_preflight": {
            str(limit): {
                "full_examples_exceeding_limit": sum(length > limit for length in full_lengths),
                "examples_with_zero_assistant_tokens_if_right_truncated": sum(
                    length >= limit for length in prefix_lengths
                ),
            }
            for limit in [512, 1024, 2048, 4096]
        },
        "actual_trainer_loss_mask_verified": False,
        "model_weights_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
