"""Pure checks for actual trainer batches; toy tests are not GPU results."""

import json
from pathlib import Path


def validate_cached_weights(snapshot):
    """Validate inference/training files, not documentation or image completeness."""
    snapshot = Path(snapshot)
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for name in required:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required cached model file missing or empty: {name}")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "cohere2":
        raise ValueError("Unexpected cached model architecture")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Invalid cached safetensors weight index")
        shards = sorted(set(mapping.values()))
    else:
        shards = ["model.safetensors"]
    for name in shards:
        if (
            not isinstance(name, str)
            or "/" in name
            or "\\" in name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Invalid weight shard filename")
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required cached weight shard missing or empty: {name}")
    return {
        "required_files_present": True,
        "weight_shards": shards,
        "total_weight_bytes": sum((snapshot / name).stat().st_size for name in shards),
        "whole_repository_snapshot_required": False,
    }


def verify_supervision(input_ids, labels, expected_boundaries):
    ids = tuple(input_ids)
    if ids not in expected_boundaries:
        raise ValueError(f"Trainer tokenization drift: unknown input sequence of {len(ids)} tokens")
    prefix = expected_boundaries[ids]
    if len(labels) != len(ids):
        raise ValueError("Labels and input tokens differ in length")
    if any(label != -100 for label in labels[:prefix]):
        raise ValueError("User/preamble tokens are incorrectly supervised")
    supervised = [index for index, label in enumerate(labels) if label != -100]
    if not supervised or any(index < prefix for index in supervised):
        raise ValueError("Empty or invalid assistant supervision")
    if any(labels[index] != ids[index] for index in supervised):
        raise ValueError("Supervised labels do not preserve the actual target tokens")
    if len(supervised) != len(ids) - prefix:
        raise ValueError("Assistant target tokens were unexpectedly masked")
    return {"input_tokens": len(ids), "prompt_tokens": prefix, "supervised_tokens": len(supervised)}
