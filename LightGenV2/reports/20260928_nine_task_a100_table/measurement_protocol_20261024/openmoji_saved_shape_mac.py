"""Derive adopted epoch45 NN MAC from already saved all600 leaf-module shapes.

Read original PT metadata only (no Torch, tensors, model or inference). Optical
dense-equivalent physics and FFT scalar work are separate, not silently folded
into this neural dense-MAC subtotal or replaced by a maximum aperture capacity.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from math import prod
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE))
from torch_metadata import read_metadata
from operation_ledger import linear_mac,convolution_mac
from prepare_openmoji_e45 import BODY
from measure_abo_text_qwen import sha

BODY_SHA='03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21'
NON_MAC={'AdaptiveAvgPool2d','AngularSpectrumPropagator','Dropout','Embedding','GELU','GroupNorm',
         'LayerNorm','PhaseLayer','ScaleOnlyCCD','Softplus'}


def module_mac(record,state):
    name=record['module'];kind=record['class_name'];output=record['outputs']
    if kind in NON_MAC:return None
    if not name.startswith('model.'):raise ValueError('Unbound non-model MAC module: '+name)
    key=name.removeprefix('model.')+'.weight';weight=state[key]['shape']
    if kind=='PositionReadout':
        if len(weight)!=1 or output['shape']!=[1,192]:raise ValueError('Original full padded64 position readout differs')
        return weight[0]*192
    inputs=record['inputs'][0]['shape']
    if kind=='Linear':return linear_mac(inputs,weight)
    if kind in ('Conv1d','Conv2d','Conv3d'):
        if len(weight)!=len(inputs) or inputs[1]%weight[1]:raise ValueError('Actual convolution channels/shape differ')
        return convolution_mac(inputs,weight,output['shape'],groups=inputs[1]//weight[1])
    raise ValueError('Unregistered observed module must not be hidden: '+kind)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    if args.output.exists():raise FileExistsError('Retain prior derived receipt')
    if sha(BODY)!=BODY_SHA:raise ValueError('Not adopted original epoch45 PT')
    with BODY.open('rb') as f:payload,raw=read_metadata(f)
    if payload['epoch']!=45:raise ValueError('Original checkpoint epoch differs')
    state=payload['model'];repeats=[];references=[];all_non_mac=Counter()
    for repeat in (1,2,3):
        folder=root/f'ours_repeat{repeat:02d}';identity=json.loads((folder/'identity.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        sealed={r['file']:r['sha256'] for r in json.loads((folder/'files_manifest.json').read_text())}
        for name in ('nn_module_shapes.json','identity.json','calls.csv','summary.json'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed actual shape/cost record changed')
        records=json.loads((folder/'nn_module_shapes.json').read_text());counts=Counter();charged=Counter();modules=Counter();non_mac=Counter()
        for row in records:
            i=row['sample_index']
            if i not in range(200):raise ValueError('Not genuine first200 shape observations')
            mac=module_mac(row,state)
            if mac is None:non_mac[row['class_name']]+=1;continue
            counts[i]+=mac;modules[row['module']]+=mac
            if not any(s in row['module'] for s in ('.language_core.blocks.','.vision_core.blocks.')):charged[i]+=mac
        if set(counts)!=set(range(200)):raise ValueError('Missing actual sample NN shapes')
        # Explicitly include actual residual NN execution in total, not charged
        # projection; non-NN FFT/phase/pooling remain separately disclosed.
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
        if len(calls)!=200:raise ValueError('Wrong cost workload')
        result=dict(repeat=repeat,actual_executed_NN_MAC200=sum(counts.values()),charged_serial_NN_MAC200=sum(charged.values()),
            parallel_residual_NN_MAC200=sum(counts.values())-sum(charged.values()),MAC_by_module=dict(modules),
            per_call=[dict(index=i,sample_id=calls[i]['sample_id'],actual_NN_MAC=counts[i],charged_serial_NN_MAC=charged[i]) for i in range(200)])
        repeats.append(result);all_non_mac.update(non_mac)
        references.append(dict(run=folder.relative_to(REPO).as_posix(),shape_sha256=sha(folder/'nn_module_shapes.json'),
            summary_sha256=sha(folder/'summary.json'),source_commit=identity['source_commit'],
            seconds200=summary['projected_seconds200'],joules200=summary['phase_estimated_j200']))
    if any(r['per_call']!=repeats[0]['per_call'] for r in repeats[1:]):raise ValueError('Original workload NN shapes vary across fresh rounds')
    actual=2*sum(r['actual_executed_NN_MAC200'] for r in repeats);serial=2*sum(r['charged_serial_NN_MAC200'] for r in repeats)
    seconds=sum(r['seconds200'] for r in references);energy=sum(r['joules200'] for r in references)
    value=dict(status='SEALED_ALL600_NN_SHAPES_AND_ORIGINAL_PT_METADATA_MAC_DERIVED',calls=600,
        checkpoint_sha256=BODY_SHA,actual_NN_MAC_OP600=actual,charged_serial_NN_MAC_OP600=serial,
        actual_NN_TOP_per_input=actual/600/1e12,charged_serial_NN_TOP_per_input=serial/600/1e12,
        actual_NN_MAC_convention_TOPS=actual/1e12/seconds,actual_NN_MAC_convention_TOPS_per_W=actual/1e12/energy,
        charged_serial_NN_MAC_convention_TOPS=serial/1e12/seconds,charged_serial_NN_MAC_convention_TOPS_per_W=serial/1e12/energy,
        repeat_results=repeats,cost_references=references,non_MAC_observed_leaf_classes=dict(all_non_mac),
        no_GPU_or_neural_forwards=True,no_new_time_or_power=True,one_real_MAC_equals_two_OP=True,
        physical_optical_dense_equivalent_OP=None,physical_optical_status='Separate physics definition required; maximum aperture capacity is not used as task OP',
        FFT_simulation_not_counted_as_physical_OP=True,complete_scalar_OP_claim=False,
        source_sha256=sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    with args.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps({k:value[k] for k in ('status','calls','actual_NN_TOP_per_input','charged_serial_NN_TOP_per_input',
        'actual_NN_MAC_convention_TOPS','actual_NN_MAC_convention_TOPS_per_W','no_GPU_or_neural_forwards')}))


if __name__=='__main__':main()
