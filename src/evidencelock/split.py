"""Connected-component split over all declared relationships, never random rows."""

import random
from collections import defaultdict

from .schemas import EvidenceCase


def relation_keys(case: EvidenceCase):
    yield ("family", case.family_id)
    yield ("template", case.template_id)
    for field in ["template_lineage", "translation_group", "paraphrase_group", "revision_group"]:
        if case.provenance.get(field):
            yield (field, case.provenance[field])


def components(cases):
    parents = {case.case_id: case.case_id for case in cases}
    if len(parents) != len(cases):
        raise ValueError("Duplicate case IDs")

    def root(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    seen = {}
    for case in cases:
        for key in relation_keys(case):
            if key in seen:
                parents[root(case.case_id)] = root(seen[key])
            else:
                seen[key] = case.case_id
    grouped = defaultdict(list)
    for case in cases:
        grouped[root(case.case_id)].append(case)
    return sorted(grouped.values(), key=lambda group: min(case.case_id for case in group))


def assign(cases, seed=42):
    groups = components(cases)
    if len(groups) < 3:
        raise ValueError(
            "At least three independent components are required for train/validation/test"
        )
    random.Random(seed).shuffle(groups)
    # Every declared lineage remains intact. A small pilot may have coarse ratios.
    train_end = max(1, min(len(groups) - 2, round(len(groups) * 0.7)))
    validation_end = max(train_end + 1, min(len(groups) - 1, round(len(groups) * 0.85)))
    result = []
    for index, group in enumerate(groups):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        result.extend(case.model_copy(update={"split": split}) for case in group)
    result.sort(key=lambda case: case.case_id)
    verify(result)
    return result


def verify(cases):
    for group in components(cases):
        if len({case.split for case in group}) != 1:
            raise ValueError("Related cases cross dataset splits")
    if any(case.split == "unassigned" for case in cases):
        raise ValueError("Dataset contains unassigned splits")
