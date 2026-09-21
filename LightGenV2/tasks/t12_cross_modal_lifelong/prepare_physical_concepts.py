"""Prepare the complete CC-BY-4.0 Physical Concepts continuity probe."""
import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


OFFICIAL_ROOT = "https://storage.googleapis.com/physical_concepts/probes"
LICENSE_PAGE = "https://github.com/google-deepmind/physical_concepts#license-and-disclaimer"
FRAME_INDEX = np.linspace(0, 14, 8).round().astype(int)


def example_message_class():
    """Build the small tf.train.Example schema with the installed protobuf."""
    fd = descriptor_pb2.FileDescriptorProto(name="tf_example.proto", package="tensorflow", syntax="proto3")
    def message(name):
        m = fd.message_type.add(); m.name = name; return m
    def field(parent, name, number, kind, label=3, type_name=None, oneof=None):
        f=parent.field.add();f.name=name;f.number=number;f.type=kind;f.label=label
        if type_name:f.type_name=type_name
        if oneof is not None:f.oneof_index=oneof
    bytes_list=message("BytesList");field(bytes_list,"value",1,12)
    float_list=message("FloatList");field(float_list,"value",1,2)
    int_list=message("Int64List");field(int_list,"value",1,3)
    feature=message("Feature");feature.oneof_decl.add().name="kind"
    field(feature,"bytes_list",1,11,label=1,type_name=".tensorflow.BytesList",oneof=0)
    field(feature,"float_list",2,11,label=1,type_name=".tensorflow.FloatList",oneof=0)
    field(feature,"int64_list",3,11,label=1,type_name=".tensorflow.Int64List",oneof=0)
    features=message("Features");entry=features.nested_type.add();entry.name="FeatureEntry";entry.options.map_entry=True
    field(entry,"key",1,9,label=1);field(entry,"value",2,11,label=1,type_name=".tensorflow.Feature")
    field(features,"feature",1,11,type_name=".tensorflow.Features.FeatureEntry")
    example=message("Example");field(example,"features",1,11,label=1,type_name=".tensorflow.Features")
    pool=descriptor_pool.DescriptorPool();pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))


EXAMPLE = example_message_class()


def records(path):
    """Read gzip-compressed TFRecord framing and decode tf.train.Example."""
    with gzip.open(path, "rb") as stream:
        while True:
            size_bytes=stream.read(8)
            if not size_bytes:return
            if len(size_bytes)!=8:raise ValueError(f"truncated TFRecord length: {path}")
            size=struct.unpack("<Q",size_bytes)[0];stream.read(4)
            payload=stream.read(size);stream.read(4)
            if len(payload)!=size:raise ValueError(f"truncated TFRecord payload: {path}")
            yield EXAMPLE.FromString(payload)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(value, namespace="select"):
    """Independent deterministic hashes for subset selection and splitting."""
    return hashlib.sha256((f"{namespace}:17:" + value).encode()).hexdigest()


def encode_video(video):
    """Eight ordered luminance frames -> 2x4 optical tile field."""
    x = torch.from_numpy(video[FRAME_INDEX].astype(np.float32) / 255.0)
    x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    x = F.interpolate(x[:, None], (112, 56), mode="bilinear", align_corners=False)[:, 0]
    field = torch.cat([torch.cat(list(x[r*4:(r+1)*4]), -1) for r in range(2)], -2)
    return (field / field.square().sum().clamp_min(1e-20).sqrt()).numpy().astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concept", default="continuity")
    p.add_argument("--max-quadruplets", type=int, default=0,
                   help="0 uses all 5,000 quadruplets; positive values are diagnostic only")
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=False)
    paths = []
    for i in range(20):
        name = f"data.tfrecord-{i:05d}-of-00020"
        path = a.cache / a.concept / name; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(f"{OFFICIAL_ROOT}/{a.concept}/{name}", path)
        paths.append(path)
    candidates = []
    for shard, path in enumerate(paths):
        for row, record in enumerate(records(path)):
            candidates.append((rank(f"{a.concept}:{shard}:{row}"), shard, row, record))
    selected = sorted(candidates, key=lambda x: x[0])
    if a.max_quadruplets:
        selected = selected[:a.max_quadruplets]
    result = {s:([], [], []) for s in ("train", "val", "test")}
    split_counts = Counter()
    for digest, shard, row, record in selected:
        identity = f"{a.concept}:{shard}:{row}"
        bucket = int(rank(identity, "split")[:8], 16) / 0xffffffff
        split = "train" if bucket < .70 else ("val" if bucket < .85 else "test")
        split_counts[split] += 1
        for kind, label in (("possible", 1), ("impossible", 0)):
            raw = record.features.feature[f"{kind}_image"].bytes_list.value[0]
            videos = np.frombuffer(raw, dtype=np.uint8).reshape(2, 15, 64, 64, 3)
            for pair in range(2):
                field = encode_video(videos[pair])
                result[split][0].append(field); result[split][1].append(label)
                result[split][2].append({"concept":a.concept,"quadruplet":f"{shard}:{row}","kind":kind,"pair":pair,"label":label})
    for split, (fields, labels, rows) in result.items():
        assert fields
        np.savez_compressed(a.out/f"{split}.npz", fields=np.stack(fields), labels=np.asarray(labels,dtype=np.int64))
        (a.out/f"{split}_records.json").write_text(json.dumps(rows,indent=2)+"\n")
    ids={s:{r["quadruplet"] for r in result[s][2]} for s in result}
    assert all(not ids[a]&ids[b] for i,a in enumerate(ids) for b in list(ids)[i+1:])
    full = len(candidates) == len(selected) == 5000
    protocol={"task":"video","classes":2,"dataset":"Physical Concepts Dataset","concept":a.concept,
              "scope":"complete continuity probe" if full else f"diagnostic subset of {len(selected)} probe quadruplets",
              "all_original_samples":full,"source_quadruplets":len(candidates),
              "input":"8 ordered luminance frames at indices "+str(FRAME_INDEX.tolist())+" tiled 2x4 into 224x224; unit total power",
              "label":{"impossible":0,"possible":1},"split":"quadruplet-disjoint hash 70/15/15",
              "quadruplets":dict(split_counts),"license":"CC BY 4.0","license_page":LICENSE_PAGE,
              "source":f"{OFFICIAL_ROOT}/{a.concept}/data.tfrecord-{{00000..00019}}-of-00020",
              "raw_shards":{p.name:sha(p) for p in paths}}
    (a.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    files={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}
    (a.out/"manifest.json").write_text(json.dumps(files,indent=2)+"\n")


if __name__ == "__main__":
    main()
