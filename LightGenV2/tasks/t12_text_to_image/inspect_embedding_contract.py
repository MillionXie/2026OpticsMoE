"""Read-only provenance and padding inspection for recovery training."""
import argparse
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("cache")
args = parser.parse_args()
payload = torch.load(args.cache, map_location="cpu", weights_only=False)
mask = payload["attention_mask"].bool()
print(json.dumps({"meta": payload["meta"], "embedding_shape": list(payload["embeddings"].shape),
                  "prompt_count": len(payload["prompts"]),
                  "length_min": int(mask.sum(1).min()), "length_max": int(mask.sum(1).max()),
                  "left_padding_rows": int((~mask[:, 0] & mask[:, -1]).sum()),
                  "first_prompt": payload["prompts"][0]}, indent=2))
