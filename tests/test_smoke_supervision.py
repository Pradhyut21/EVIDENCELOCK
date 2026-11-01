import pytest

from gpu.smoke_checks import verify_supervision


def test_exact_assistant_mask():
    assert verify_supervision([1, 2, 3, 4], [-100, -100, 3, 4], {(1, 2, 3, 4): 2}) == {
        "input_tokens": 4,
        "prompt_tokens": 2,
        "supervised_tokens": 2,
    }


@pytest.mark.parametrize(
    "labels", [[1, 2, 3, 4], [-100, -100, -100, -100], [-100, -100, 99, 4], [-100, -100, 3, -100]]
)
def test_invalid_masks_fail_closed(labels):
    with pytest.raises(ValueError):
        verify_supervision([1, 2, 3, 4], labels, {(1, 2, 3, 4): 2})


def test_tokenizer_drift_is_not_silently_accepted():
    with pytest.raises(ValueError, match="drift"):
        verify_supervision([1, 2, 3, 5], [-100, -100, 3, 5], {(1, 2, 3, 4): 2})
