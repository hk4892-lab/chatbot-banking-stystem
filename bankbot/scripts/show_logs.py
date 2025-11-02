"""Pretty-print the last N audit log entries."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.log"


def tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    buffer: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            buffer.append(line.rstrip("\n"))
    return list(buffer)


def pretty_print(entries: list[str]) -> None:
    for line in entries:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(line)
            continue
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show recent audit log entries")
    parser.add_argument("--tail", type=int, default=20, help="Number of entries to display")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Audit log path")
    args = parser.parse_args()

    entries = tail_lines(args.path, args.tail)
    if not entries:
        print("No log entries found.")
        return
    pretty_print(entries)


if __name__ == "__main__":
    main()
