"""Package a narrowed latent editor and its trained Qwen-mini head as one weight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def pack(*, editor_checkpoint: Path, bridge_checkpoint: Path, output: Path,
         counted_parameters: int) -> dict:
    editor = torch.load(editor_checkpoint, map_location="cpu", weights_only=False)
    bridge = torch.load(bridge_checkpoint, map_location="cpu", weights_only=False)
    if int(bridge["meta"]["counted_text_parameters"]) < 1:
        raise ValueError("Missing counted Qwen-mini parameters")
    editor["text_frontend"] = {
        "kind": "qwen-mini-two-block-640-plus-bridge",
        "config": bridge["text_config"],
        "text": bridge["text"],
        "bridge": bridge["bridge"],
        "counted_parameters": bridge["meta"]["counted_text_parameters"],
        "shared_token_embedding_excluded": True,
    }
    editor["counted_parameters"] = counted_parameters
    editor["unified_modes"] = ["background", "object", "joint"]
    editor["generator_calls"] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(editor, output)
    return {"output": str(output), "counted_parameters": counted_parameters,
            "qwen_layers": bridge["text_config"]["layers"],
            "text_frontend_parameters": bridge["meta"]["counted_text_parameters"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("editor-checkpoint","bridge-checkpoint","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--counted-parameters",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(pack(editor_checkpoint=args.editor_checkpoint.resolve(),
                          bridge_checkpoint=args.bridge_checkpoint.resolve(),
                          output=args.output.resolve(),
                          counted_parameters=args.counted_parameters),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
