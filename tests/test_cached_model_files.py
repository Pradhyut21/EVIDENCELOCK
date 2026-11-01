import json

import pytest

from gpu.smoke_checks import validate_cached_weights


def configuration(root):
    (root / "config.json").write_text(json.dumps({"model_type": "cohere2"}))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")


def test_readme_and_images_are_not_required(tmp_path):
    configuration(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"cache-presence-test-fixture")
    result = validate_cached_weights(tmp_path)
    assert result["required_files_present"] is True
    assert not result["whole_repository_snapshot_required"]


def test_every_indexed_shard_is_required(tmp_path):
    configuration(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.a": "part1.safetensors", "layer.b": "part2.safetensors"}})
    )
    (tmp_path / "part1.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="part2"):
        validate_cached_weights(tmp_path)
    (tmp_path / "part2.safetensors").write_bytes(b"fixture")
    assert len(validate_cached_weights(tmp_path)["weight_shards"]) == 2


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_missing_required_file_is_not_ignored(tmp_path, missing):
    configuration(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_cached_weights(tmp_path)


def test_empty_weight_file_rejected(tmp_path):
    configuration(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_cached_weights(tmp_path)


def test_weight_index_cannot_escape_snapshot(tmp_path):
    configuration(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.a": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="filename"):
        validate_cached_weights(tmp_path)
