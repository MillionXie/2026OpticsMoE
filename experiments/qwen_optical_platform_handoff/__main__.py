from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_and_validate_contract


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a simulation or hardware Qwen-optical task contract."
    )
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    contract = load_and_validate_contract(args.contract)
    print(
        json.dumps(
            {
                "status": "valid",
                "contract": str(args.contract.resolve()),
                "project": contract["project"]["name"],
                "task_kind": contract["task"]["kind"],
                "hardware_enabled": contract["hardware"]["enabled"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
