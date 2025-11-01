"""Pretty-print the last N audit log entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    return lines[-count:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show recent audit log entries")
    parser.add_argument("--tail", type=int, default=20, help="Number of entries to display")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "audit.log",
        help="Override log file path",
    )
    args = parser.parse_args()

    lines = tail_lines(args.log_path, args.tail)
    if not lines:
        print("No log entries found.")
        return

    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(line.strip())
            continue
        ts = payload.get("ts")
        route = payload.get("route")
        lang = payload.get("lang")
        topdoc = payload.get("topdoc_id")
        score = payload.get("top_score")
        tools = ", ".join(payload.get("tools_called", [])) or "-"
        tokens = ", ".join(payload.get("redaction_tokens", [])) or "-"
        print(f"[{ts}] lang={lang} route={route} topdoc={topdoc} score={score}")
        print(f"  tools={tools}")
        print(f"  tokens={tokens}")


if __name__ == "__main__":
    main()
