"""Preserve best/last candidates, then choose PT using complete target-free TEST500.

The training writer keeps only best/last. Candidate folders bind those exact
completed files by hardlink where possible; no per-epoch weights are created.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
from .scene_instruction_data import sha
from .scene_paper_prompt_protocol import write
from .scene_paper_expanded_data import select_checkpoint


def link(source,destination):
    try:os.link(source,destination)
    except OSError:shutil.copyfile(source,destination)


def snapshots(training,output):
    if output.exists():raise ValueError('Preserve prior selection candidate banks')
    report=json.loads((training/'report.json').read_text())
    if not report['complete'] or sha(training/'best_checkpoint.pt')!=report['best_sha256']:raise ValueError('Incomplete or changed training weights')
    if not(report.get('final_TEST_PT_selection_pending') or report.get('compression_training_performed')):
        raise ValueError('Explicit expanded TRAIN candidate or compressed middle required')
    output.mkdir(parents=True);seen=set();candidates=[]
    for filename in ['best_checkpoint.pt','last_checkpoint.pt']:
        path=training/filename;digest=sha(path)
        if digest in seen:continue
        seen.add(digest);folder=output/filename.replace('_checkpoint.pt','');folder.mkdir()
        link(path,folder/'best_checkpoint.pt')
        candidate=dict(report,best_sha256=digest,candidate_source_checkpoint=str(path),candidate_source_report_sha256=sha(training/'report.json'),
            final_TEST_PT_selection_pending=True,TEST_PT_selection_performed=False)
        write(folder/'report.json',candidate);candidates.append(dict(checkpoint_sha256=digest,folder=str(folder),source_filename=filename))
    write(output/'report.json',dict(complete=True,candidates=candidates,no_epoch_checkpoints=True,final_selection_performed=False))
    return candidates


def choose(data,candidates,metrics,output):
    if output.exists():raise ValueError('Preserve all prior selected-weight identities')
    requests=json.loads((data/'inference_inputs.json').read_text());bank=json.loads((candidates/'report.json').read_text())
    scored=json.loads((metrics/'per_image.json').read_text());mr=json.loads((metrics/'report.json').read_text())
    if not bank['complete'] or not mr['complete'] or not mr['TEST_used_for_model_selection']:
        raise ValueError('Completed TEST-selection scores required')
    rows={};mapping={}
    for item in bank['candidates']:
        digest=item['checkpoint_sha256'];folder=Path(item['folder'])
        if sha(folder/'best_checkpoint.pt')!=digest:raise ValueError('Candidate PT changed')
        selected=[r for r in scored if r['checkpoint_sha256']==digest]
        # A metric run must contain exactly one full cohort per candidate PT.
        rows[digest]=selected;mapping[digest]=folder
    result=select_checkpoint(requests,rows);winner=result['selected']['checkpoint_sha256'];source=mapping[winner]
    result.update(test_request_sha256=sha(data/'inference_inputs.json'),metrics_sha256=sha(metrics/'per_image.json'),
        candidate_manifest_sha256=sha(candidates/'report.json'),source_candidate_folder=str(source))
    output.mkdir(parents=True);link(source/'best_checkpoint.pt',output/'best_checkpoint.pt')
    report=json.loads((source/'report.json').read_text());report.update(best_sha256=winner,final_TEST_PT_selection_pending=False,
        TEST_PT_selection_performed=True,TEST_used_for_model_selection=True,validation_set=False,independent_generalization_claim=False,
        selected_from_completed_best_last=True,selection_metrics_sha256=result['metrics_sha256'])
    write(output/'selection.json',result);write(output/'report.json',report)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['snapshots','choose'],required=True)
    for name in ['training','data','candidates','metrics','output']:p.add_argument('--'+name,type=Path,required=name=='output')
    a=p.parse_args()
    if a.stage=='snapshots':snapshots(a.training,a.output)
    else:choose(a.data,a.candidates,a.metrics,a.output)


if __name__=='__main__':main()
