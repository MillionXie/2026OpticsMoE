"""Find the best CCD dihedral orientation independently for every physical stage."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


STAGES = ("vision_router", "vision_expert", "vision_global", "language_router", "language_expert", "language_global")


def variants(a):
    return {"identity": a, "flip_h": np.fliplr(a), "flip_v": np.flipud(a), "rot180": np.rot90(a, 2),
            "transpose": a.T, "rot90": np.rot90(a, 1), "rot270": np.rot90(a, 3), "anti_transpose": np.rot90(a.T, 2)}


def orient(a, name): return variants(a)[name].copy()


def pcc(a, b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();a-=a.mean();b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b);return float(a.dot(b)/d) if d else 0.0


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args()
    report=json.loads(a.report.read_text(encoding="utf-8"));ids=[x["sample_id"] for x in report["samples"]]
    ref=np.load(a.root/"simulation_reference.npz");saved=report["camera_canonical_orientation"]
    rows={}
    for stage in STAGES:
        # Undo the orientation applied before PNG saving, then test all candidates.
        captured=[orient(np.asarray(Image.open(a.root/"ccd"/stage/f"{sid}.png"),np.float32),saved) for sid in ids]
        scores=[]
        for name in variants(captured[0]):
            values=[pcc(orient(x,name),y) for x,y in zip(captured,ref[stage])]
            scores.append({"orientation":name,"mean_pcc":float(np.mean(values)),"per_sample":values})
        rows[stage]=sorted(scores,key=lambda x:x["mean_pcc"],reverse=True)[:3]
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
