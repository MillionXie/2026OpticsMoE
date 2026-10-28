"""Two-card queue for the authorized N16/k8 three-seed TRAIN-only follow-up.

Reuse the established continuation/evaluation implementations. Never overlap
one of our own GPU workers, and never interrupt a foreign process.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .expert_tail_schedule import audit_source,save,sha


def available(free_mib,commands,own_prefixes):
    return free_mib>=6000 and not any(prefix in command for command in commands for prefix in own_prefixes)


def exceeds_previous(counts):
    assert len(counts)==3 and all(type(v) is int and 0<=v<=752 for v in counts)
    return sum(counts)>1588


def validate(c):
    assert c['seeds']==[17,27,37] and sorted(j['seed'] for j in c['jobs'])==c['seeds']
    assert all(j['experts']==16 and j['top_k']==8 and j['source_epochs']==60 for j in c['jobs'])
    assert c['extra_epochs']==30 and c['lr']==.0005 and c['microbatch']==2
    assert c['margin_weight']==.2 and c['capture_weight']==.005 and c['minimum_free_mib']==6000
    assert c['gpus']=={'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}
    assert c['baseline_counts']==[522,518,540] and c['previous_k4_counts']==[541,512,535]
    assert c['minimum_total_correct']==1589 and c['TEST_enters_gradient'] is False


def command(pid):
    p=Path('/proc',str(pid),'cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode(errors='replace') if p.exists() else ''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();c=json.loads(a.profile.read_text());validate(c)
    commit=os.environ['T10_N16_MIDK_SOURCE_COMMIT']
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip());task=Path(__file__).parent
    files=[Path(__file__),a.profile,*[task/name for name in ['diagnose_topk.py','expert_tail_schedule.py','train.py','model.py','plan.py','evaluate_test.py','locked_test_cache.py','configs/study.json']],repo/'LightGenV2/common/optical_contract.py']
    hashes={}
    for p in files:
        key=p.resolve().relative_to(repo).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+key]);hashes[key]=sha(p)
    audits=[audit_source(j) for j in c['jobs']]
    assert all(r['data_sha256']==c['data_sha256'] for r in audits)
    a.out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(j,out=str(a.out/f'N16_k8_s{j["seed"]}')) for j in c['jobs']]
    save(a.out/'plan.json',dict(configuration=c,published_source_commit=commit,source_files=hashes,profile_sha256=sha(a.profile),
        command=sys.argv,pid=os.getpid(),source_audits=audits,host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol='TRAIN-only30 full epochs, retained live/Adam/EMA, VAL accuracy/NLL selection including old best; all3 models locked before TEST; post-observed-TEST development follow-up.'))
    pending=list(jobs);running={};trained=[];rows=[];failures={};pids=[];phase='training';lock=a.out/'all_models_test_lock.json'
    prefixes=c['blocked_own_run_prefixes']+[str(a.out)]
    try:
        while pending or running:
            for gpu,(proc,log,job) in list(running.items()):
                if proc.poll() is None:continue
                proc.wait();log.close();del running[gpu]
                if proc.returncode:
                    failures[str(job['seed'])]=dict(phase=phase,returncode=proc.returncode,pid=proc.pid,out=job['out']);continue
                out=Path(job['out'])
                if phase=='training':
                    r=json.loads((out/'result.json').read_text());assert sha(out/'best_checkpoint.pt')==r['checkpoint_sha256']
                    assert json.loads((out/'status.json').read_text())['state']=='validation_complete_test_deferred'
                    trained.append(job)
                else:
                    r=json.loads((out/'result.json').read_text());t=json.loads((out/'test_result.json').read_text())
                    assert sha(out/'best_checkpoint.pt')==r['checkpoint_sha256']==t['checkpoint_sha256']
                    correct=sum(t['test']['confusion_matrix'][i][i] for i in range(8));assert sum(map(sum,t['test']['confusion_matrix']))==752
                    row=dict(seed=job['seed'],experts=16,top_k=8,source=job['source'],run=job['out'],source_best_sha256=job['best_sha256'],source_last_sha256=job['last_sha256'],
                        checkpoint_sha256=r['checkpoint_sha256'],last_checkpoint_sha256=sha(out/'last_checkpoint.pt'),best_epoch=r['best_epoch'],total_epochs=90,
                        train=r['train'],val=r['val'],test=t['test'],test_correct=int(correct),test_samples=752,TEST_gradient_inputs=0,
                        original_TRAIN_gradient_inputs=30*3496,extra_optimizer_updates=30*219,selection='Locked validation accuracy, NLL tie, including old best')
                    rows.append(row);save(out/'followup_completion.json',row)
            if not pending and not running and phase=='training':
                if failures:break
                assert len(trained)==3
                save(lock,dict(selection='validation_accuracy_NLL_tie_including_original_best',all_models_locked_before_any_TEST=True,
                    role='Development follow-up after original TEST was observed; TEST not in gradients',
                    runs=[dict(run_dir=str(Path(j['out']).resolve()),checkpoint_sha256=sha(Path(j['out'])/'best_checkpoint.pt')) for j in jobs]))
                phase='testing';pending=list(jobs)
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
            by_uuid={uuid:[] for uuid in c['gpus'].values()}
            for line in compute.splitlines():
                if not line.strip():continue
                pid,uuid=[s.strip() for s in line.split(',',1)]
                if uuid in by_uuid:by_uuid[uuid].append(command(int(pid)))
            for text,uuid in c['gpus'].items():
                gpu=int(text)
                if not pending or gpu in running:continue
                actual=subprocess.check_output(['nvidia-smi','-i',text,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                assert actual[0].strip()==uuid
                blockers=[command(b['pid']) for b in c['blocked_processes'] if b['gpu']==gpu and b['run_prefix'] in command(b['pid'])]
                if not available(int(actual[1]),by_uuid[uuid]+blockers,prefixes):continue
                job=pending.pop(0);out=Path(job['out'])
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',T10_SOURCE_COMMIT=commit,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                if phase=='training':
                    assert not out.exists()
                    cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.diagnose_topk','--source',job['source'],'--out',str(out),
                        '--epochs','30','--lr','0.0005','--batch','2','--margin-weight','0.2','--capture-weight','0.005','--selection-metric','accuracy','--route-audit','--defer-test']
                else:
                    metadata=json.loads((out/'metadata.json').read_text())
                    cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test','--run',str(out),'--data',metadata['arguments']['data'],
                        '--selection-lock',str(lock),'--batch','2','--cache-locked-result']
                log=(a.out/f'seed{job["seed"]}_{phase}.log').open('x')
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(child,log,job);pids.append(child.pid)
            save(a.out/'status.json',dict(state=phase,pid=os.getpid(),pending_seeds=[j['seed'] for j in pending],trained_seeds=[j['seed'] for j in trained],
                tested_seeds=[j['seed'] for j in rows],running={str(gpu):dict(pid=p.pid,seed=j['seed'],out=j['out']) for gpu,(p,_,j) in running.items()},failures=failures,timestamp=time.time()))
            if pending or running:time.sleep(10)
        if not failures:
            rows.sort(key=lambda j:j['seed']);assert [j['seed'] for j in rows]==[17,27,37]
            counts=[j['test_correct'] for j in rows];values=[100*n/752 for n in counts]
            save(a.out/'summary.json',dict(rows=rows,correct_counts=counts,total_correct=sum(counts),mean_percent=statistics.mean(values),
                sample_sd_percent=statistics.stdev(values),minimum_target_total=1589,strict_target_met=exceeds_previous(counts),
                role='TRAIN-only post-observed-TEST development follow-up; added30 full epochs per seed, not original equal-budget evidence'))
        save(a.out/'status.json',dict(state='failed' if failures else 'complete',pid=os.getpid(),running={},failures=failures,pending_seeds=[],timestamp=time.time()))
    except BaseException as error:
        save(a.out/'status.json',dict(state='scheduler_error',pid=os.getpid(),error=repr(error),running={str(g):dict(pid=p.pid,seed=j['seed'],out=j['out']) for g,(p,_,j) in running.items()},
            note='Preserve child evidence; no foreign process stopped',timestamp=time.time()));raise
    finally:
        save(a.out/'release_check.json',dict(child_pids=pids,own_children_exited=all(not command(pid) or str(a.out) not in command(pid) for pid in pids),timestamp=time.time(),foreign_untouched=True))


if __name__=='__main__':main()
