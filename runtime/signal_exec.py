#!/usr/bin/env python3
"""Exec a command after clearing termination signals inherited from a parent mask."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Sequence


def unblocked_exec_command(command: Sequence[str]) -> list[str]:
    """Wrap an argv for use while the launching parent has INT/TERM blocked."""
    values = [str(value) for value in command]
    if not values:
        raise ValueError("child command must not be empty")
    if not hasattr(signal, "pthread_sigmask"):
        return values
    return [sys.executable, str(Path(__file__).resolve()), "--", *values]


def main(argv: Sequence[str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("signal_exec: a command is required", file=sys.stderr)
        return 2
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
