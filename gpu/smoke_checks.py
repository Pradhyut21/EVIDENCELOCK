"""Pure checks for actual trainer batches; toy tests are not GPU results."""


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
