from __future__ import annotations

import json
from typing import Any


def dumps_strict_json(value: Any, *, ensure_ascii: bool = False, sort_keys: bool = True) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=sort_keys, allow_nan=False)


def loads_strict_json(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
