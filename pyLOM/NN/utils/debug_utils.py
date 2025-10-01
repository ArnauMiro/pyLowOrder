"""Shared utilities for debug logging across NN components.

This module centralises logic for configuring debug verbosity and exposes a
simple helper to translate user-friendly inputs (bool, str, int) into a common
`DebugLevel` enum.  Components can import this to keep behaviour consistent.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, Union


class DebugLevel(IntEnum):
    """Ordered debug verbosity levels used by NN components.

    Levels are intentionally compact so they can be compared using simple
    integer relations.  Higher numbers imply more verbose output.
    """

    OFF = 0
    BASIC = 1
    DETAILED = 2
    TRACE = 3


_LEVEL_ALIASES = {
    "off": DebugLevel.OFF,
    "false": DebugLevel.OFF,
    "none": DebugLevel.OFF,
    "basic": DebugLevel.BASIC,
    "light": DebugLevel.BASIC,
    "minimal": DebugLevel.BASIC,
    "detailed": DebugLevel.DETAILED,
    "verbose": DebugLevel.DETAILED,
    "advanced": DebugLevel.DETAILED,
    "trace": DebugLevel.TRACE,
    "debug": DebugLevel.TRACE,
    "insane": DebugLevel.TRACE,
    "true": DebugLevel.BASIC,
}


def _normalise_input(value: Union[DebugLevel, str, int, bool, None]) -> Optional[DebugLevel]:
    if value is None:
        return None
    if isinstance(value, DebugLevel):
        return value
    if isinstance(value, bool):
        return DebugLevel.BASIC if value else DebugLevel.OFF
    if isinstance(value, int):
        try:
            return DebugLevel(value)
        except ValueError:
            return DebugLevel.BASIC if value > 0 else DebugLevel.OFF
    if isinstance(value, str):
        key = value.strip().lower()
        if key.isdigit():
            return _normalise_input(int(key))
        return _LEVEL_ALIASES.get(key, DebugLevel.BASIC)
    return None


def resolve_debug_level(*values: Union[DebugLevel, str, int, bool, None], default: DebugLevel = DebugLevel.OFF) -> DebugLevel:
    """Collapse a sequence of user-provided debug hints into a `DebugLevel`.

    The first value that can be interpreted is used.  If none can be resolved
    the provided `default` is returned.
    """

    for value in values:
        level = _normalise_input(value)
        if level is not None:
            return level
    return default


def level_enabled(current: Union[DebugLevel, int], required: Union[DebugLevel, int]) -> bool:
    """Return True if a message at ``required`` should be emitted."""

    return int(current) >= int(required)


def format_debug_prefix(component: str, level: Union[DebugLevel, int]) -> str:
    """Lightweight prefix helper to make multi-component logs easier to parse."""

    return f"[{component}@L{int(level)}]"


def unpack_dprint_args(args: Any) -> Any:
    """Utility that keeps backwards compatibility with legacy `_dprint` calls."""

    # Currently a trivial passthrough, but centralised to ease future changes.
    return args
