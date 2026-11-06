"""Lossless observable JSON codec; no evidence policy or oracle is evaluated.

EL-OBS-1 uses h for the unchanged header, c for evidence column names once,
and r for rows. Optional p contains presence bitmasks when records have different
keys. A missing field is therefore distinct from an explicit JSON null.
JSON object key order is not significant; list order and all values are preserved.
"""
import json
import math

FORMAT = 'EL-OBS-1'
OBSERVABLE_KEYS = {'question', 'evidence_items', 'as_of', 'query_policy', 'scope_context'}


def _validate_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError('JSON object keys must be strings')
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError('Expected JSON values only; nonfinite numbers are forbidden')


def _validate_observable(payload):
    _validate_json(payload)
    if type(payload) is not dict or set(payload) != OBSERVABLE_KEYS:
        raise ValueError('Encode the explicitly selected observable_payload, not a full case')
    if type(payload['query_policy']) is not dict or 'in_scope' in payload['query_policy']:
        raise ValueError('query_policy must be a JSON object without the hidden in_scope annotation')
    if type(payload['scope_context']) is not dict:
        raise ValueError('scope_context must be a JSON object')
    if type(payload['evidence_items']) is not list or any(type(r) is not dict for r in payload['evidence_items']):
        raise ValueError('evidence_items must be an array of JSON objects')


def encode_observables(payload):
    """Return compact JSON; raw content is quoted data and is never executed."""
    _validate_observable(payload)
    records = payload['evidence_items']
    columns = sorted({key for record in records for key in record})
    header = {key: value for key, value in payload.items() if key != 'evidence_items'}
    complete = all(set(record) == set(columns) for record in records)
    packed = {
        'v': FORMAT,
        'h': header,
        'c': columns,
        'r': [[record[key] for key in columns if key in record] for record in records],
    }
    if not complete:
        packed['p'] = [sum(1 << index for index, key in enumerate(columns) if key in record)
                       for record in records]
    return json.dumps(packed, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object keys are forbidden')
        result[key] = value
    return result


def decode_observables(encoded):
    """Reconstruct the exact observable JSON value, including missing/null fields."""
    if type(encoded) is not str:
        raise ValueError('Encoded ledger must be a JSON string')
    packed = json.loads(encoded, object_pairs_hook=_unique_object)
    _validate_json(packed)
    if type(packed) is not dict or set(packed) not in ({'v', 'h', 'c', 'r'}, {'v', 'h', 'c', 'r', 'p'}):
        raise ValueError('Invalid encoded ledger structure')
    if packed['v'] != FORMAT:
        raise ValueError('Unsupported observable ledger version')
    header, columns, rows = packed['h'], packed['c'], packed['r']
    if type(header) is not dict or set(header) != OBSERVABLE_KEYS - {'evidence_items'}:
        raise ValueError('Invalid observable header')
    if type(columns) is not list or not all(type(key) is str for key in columns) or len(set(columns)) != len(columns):
        raise ValueError('Invalid evidence columns')
    if type(rows) is not list or any(type(row) is not list for row in rows):
        raise ValueError('Invalid evidence rows')
    masks = packed.get('p', [(1 << len(columns)) - 1] * len(rows))
    if type(masks) is not list or len(masks) != len(rows):
        raise ValueError('Invalid presence mask count')
    records = []
    for row, mask in zip(rows, masks, strict=True):
        if type(mask) is not int or not 0 <= mask < (1 << len(columns)):
            raise ValueError('Invalid presence bitmask')
        present = [key for index, key in enumerate(columns) if mask & (1 << index)]
        if len(present) != len(row):
            raise ValueError('Evidence row does not match its declared columns')
        records.append(dict(zip(present, row, strict=True)))
    result = {**header, 'evidence_items': records}
    _validate_observable(result)
    return result
