import hashlib
import json

import pytest

from gpu.final_driver import verify_freeze


@pytest.fixture
def frozen(tmp_path):
    directory = tmp_path / "final-evaluation"
    directory.mkdir()
    weight = directory / "adapter.safetensors"
    weight.write_bytes(b"synthetic fixture, never a trained model")
    manifest = {
        "training_enabled": False,
        "test_used_for_selection": False,
        "files_sha256": {
            "final-evaluation/adapter.safetensors": hashlib.sha256(weight.read_bytes()).hexdigest()
        },
    }
    (directory / "freeze.json").write_text(json.dumps(manifest))
    return tmp_path, directory, weight, manifest


def test_verification_needs_no_gpu_or_ml_packages(frozen):
    root, _, _, manifest = frozen
    assert verify_freeze(root) == manifest


def test_changed_frozen_adapter_is_rejected(frozen):
    root, _, weight, _ = frozen
    weight.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_freeze(root)


def test_missing_frozen_adapter_is_rejected(frozen):
    root, _, weight, _ = frozen
    weight.unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        verify_freeze(root)


@pytest.mark.parametrize("flag", ["training_enabled", "test_used_for_selection"])
def test_training_or_test_driven_selection_is_rejected(frozen, flag):
    root, directory, _, manifest = frozen
    manifest[flag] = True
    (directory / "freeze.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot train or select"):
        verify_freeze(root)


def test_manifest_cannot_read_outside_its_bundle(frozen):
    root, directory, _, manifest = frozen
    manifest["files_sha256"] = {"../unrelated": "0" * 64}
    (directory / "freeze.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing or unsafe"):
        verify_freeze(root)
