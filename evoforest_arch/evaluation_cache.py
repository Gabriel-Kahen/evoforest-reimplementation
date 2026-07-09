from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, fields, is_dataclass
import copy
import hashlib
import math
import pickle
import threading
import types
from typing import Any

import numpy as np


_CACHE_KEY_VERSION = "evoforest-evaluation-cache-v1"


def fingerprint_inputs(inputs: dict[str, object]) -> str:
    """Return a content fingerprint suitable for isolating evaluation datasets."""
    digest = hashlib.blake2b(digest_size=20)
    digest.update(_CACHE_KEY_VERSION.encode("utf-8"))
    _update_digest(digest, inputs)
    return digest.hexdigest()


def fingerprint_value(value: object) -> str:
    digest = hashlib.blake2b(digest_size=20)
    _update_digest(digest, value)
    return digest.hexdigest()


def fingerprint_callable(fn: object) -> str:
    """Fingerprint executable semantics while ignoring incidental mutable counters."""
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(getattr(fn, "__module__", type(fn).__module__)).encode("utf-8"))
    digest.update(str(getattr(fn, "__qualname__", type(fn).__qualname__)).encode("utf-8"))
    code = getattr(fn, "__code__", None)
    if code is not None:
        digest.update(code.co_code)
        _update_digest(digest, code.co_consts)
        _update_digest(digest, code.co_names)
    _update_digest(digest, getattr(fn, "__defaults__", None))
    _update_digest(digest, getattr(fn, "__kwdefaults__", None))
    closure = getattr(fn, "__closure__", None) or ()
    freevars = code.co_freevars if code is not None else ()
    ignored_runtime_state = {"calls", "counts", "counter", "stats", "metrics", "local_callable", "torch_callable"}
    for name, cell in zip(freevars, closure, strict=False):
        try:
            captured = cell.cell_contents
        except ValueError:
            continue
        # Test/observability counters and lazily compiled source callables are
        # operational state rather than executable semantics. Other mutable
        # closure values are included so parameter changes invalidate the path.
        if name in ignored_runtime_state:
            continue
        if callable(captured):
            digest.update(str(getattr(captured, "__module__", type(captured).__module__)).encode("utf-8"))
            digest.update(str(getattr(captured, "__qualname__", type(captured).__qualname__)).encode("utf-8"))
            captured_code = getattr(captured, "__code__", None)
            if captured_code is not None:
                _update_digest(digest, captured_code)
        else:
            _update_digest(digest, captured)
    return digest.hexdigest()


def _update_digest(digest: Any, value: object) -> None:
    if isinstance(value, types.CodeType):
        digest.update(b"code\0")
        digest.update(value.co_code)
        _update_digest(digest, value.co_consts)
        _update_digest(digest, value.co_names)
        _update_digest(digest, value.co_varnames)
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        if array.dtype.hasobject:
            digest.update(pickle.dumps(array.tolist(), protocol=5))
        else:
            contiguous = np.ascontiguousarray(array)
            digest.update(memoryview(contiguous).cast("B"))
        return
    if isinstance(value, np.generic):
        _update_digest(digest, value.item())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, (set, frozenset)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in sorted((fingerprint_value(item) for item in value)):
            digest.update(item.encode("ascii"))
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(value)
        return
    if isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
        return
    if isinstance(value, float):
        digest.update(b"float\0")
        digest.update(("nan" if math.isnan(value) else value.hex()).encode("ascii"))
        return
    if isinstance(value, (bool, int, complex, type(None))):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(repr(value).encode("ascii"))
        return
    if is_dataclass(value):
        digest.update(f"dataclass:{type(value).__module__}.{type(value).__qualname__}\0".encode("utf-8"))
        for item in fields(value):
            digest.update(item.name.encode("utf-8"))
            _update_digest(digest, getattr(value, item.name))
        return
    try:
        payload = pickle.dumps(value, protocol=5)
    except Exception as exc:  # pragma: no cover - defensive path for external loaders
        raise TypeError(f"Cannot construct a stable evaluation-cache fingerprint for {type(value).__name__}.") from exc
    digest.update(f"pickle:{type(value).__module__}.{type(value).__qualname__}\0".encode("utf-8"))
    digest.update(payload)


@dataclass
class _CacheEntry:
    value: object
    nbytes: int
    epoch: int


class PersistentEvaluationCache(MutableMapping[object, object]):
    """A bounded, process-local LRU cache for immutable evaluation snapshots."""

    def __init__(self, max_entries: int = 4096, max_bytes: int = 1_073_741_824) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self._entries: OrderedDict[object, _CacheEntry] = OrderedDict()
        self._bytes = 0
        self._epoch = 0
        self._hits = 0
        self._cross_evaluation_hits = 0
        self._stores = 0
        self._evictions = 0
        self._lock = threading.RLock()

    def begin_evaluation(self) -> None:
        with self._lock:
            self._epoch += 1

    def __getitem__(self, key: object) -> object:
        with self._lock:
            entry = self._entries[key]
            self._entries.move_to_end(key)
            self._hits += 1
            if entry.epoch < self._epoch:
                self._cross_evaluation_hits += 1
            return _borrow_snapshot(entry.value)

    def __setitem__(self, key: object, value: object) -> None:
        snapshot = _freeze_snapshot(copy.deepcopy(value))
        nbytes = _estimate_nbytes(snapshot)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.nbytes
            self._entries[key] = _CacheEntry(snapshot, nbytes, self._epoch)
            self._bytes += nbytes
            self._stores += 1
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _old_key, old_entry = self._entries.popitem(last=False)
                self._bytes -= old_entry.nbytes
                self._evictions += 1

    def __delitem__(self, key: object) -> None:
        with self._lock:
            entry = self._entries.pop(key)
            self._bytes -= entry.nbytes

    def __iter__(self) -> Iterator[object]:
        with self._lock:
            return iter(tuple(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": int(self._bytes),
                "hits": int(self._hits),
                "cross_evaluation_hits": int(self._cross_evaluation_hits),
                "stores": int(self._stores),
                "evictions": int(self._evictions),
                "epoch": int(self._epoch),
                "max_entries": int(self.max_entries),
                "max_bytes": int(self.max_bytes),
            }


def _freeze_snapshot(value: object) -> object:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _freeze_snapshot(item)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _freeze_snapshot(item)
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_snapshot(item) for item in value)
    if is_dataclass(value):
        for item in fields(value):
            object.__setattr__(value, item.name, _freeze_snapshot(getattr(value, item.name)))
    return value


def _borrow_snapshot(value: object) -> object:
    if isinstance(value, np.ndarray):
        borrowed = value.view()
        borrowed.setflags(write=False)
        return borrowed
    if isinstance(value, dict):
        return {key: _borrow_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_borrow_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_borrow_snapshot(item) for item in value)
    if is_dataclass(value):
        borrowed = copy.copy(value)
        for item in fields(value):
            object.__setattr__(borrowed, item.name, _borrow_snapshot(getattr(value, item.name)))
        return borrowed
    return value


def _estimate_nbytes(value: object, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_estimate_nbytes(key, seen) + _estimate_nbytes(item, seen) for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_estimate_nbytes(item, seen) for item in value)
    if is_dataclass(value):
        return sum(_estimate_nbytes(getattr(value, item.name), seen) for item in fields(value))
    return 0
