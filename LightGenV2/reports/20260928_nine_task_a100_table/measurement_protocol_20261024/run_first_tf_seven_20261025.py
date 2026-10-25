"""Human-authorized seven Qwen baselines only, original 200 x 3 fresh workers."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from first_tf_supplement_20261025 import sha,write
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
def templates():
    return [
      ('02','t08_abo_image_text_retrieval','a100_image_input_inclusive_20261024_02','vision'),
      ('03','t07_abo_image_retrieval','a100_input_inclusive_20261024_qwen01','vision'),
      ('04','t08_abo_image_text_retrieval','a100_title_input_inclusive_20261024_02','language'),
      ('05','t02_keypoint_detection','a100_input_inclusive_20261024_02','vision'),
      ('06','t03_saliency','a100_input_inclusive_20261024_01','vision'),
      ('07','t04_semantic_interaction','a100_qwen_input_inclusive_20261025_01','vision'),
      ('08','t12_text_to_image','a100_gpu_checked_remaining_20261025_01','language')]
def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation') or not 0<args.deadline-time.time()<=21600:raise ValueError('New bounded suite required')
    root.mkdir(parents=True);child=None;active=False
    sources={str(f.relative_to(REPO)):sha(f) for f in HERE.glob('*.py') if f.name in
        ('first_tf_supplement_20261025.py','run_first_tf_seven_20261025.py','run_temporal_remainder.py','stage_meter.py',
         'measure_abo_image_qwen.py','measure_abo_text_qwen.py','measure_qwen.py','measure_lsp.py','measure_salicon.py',
         'measure_openmoji_qwen.py','measure_t12_cost.py','gpu_checked_cost_entry_20261025.py')}
    state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,completed=[],sources=sources,
        command=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        authorization='Only seven missing first-TF baseline time/energy; retained LG, proofs, TEST and other baselines not rerun')
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-30:raise TimeoutError('Finite queue deadline')
        for f,d in sources.items():
            if sha(REPO/f)!=d:raise ValueError('Frozen source changed: '+f)
    save()
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import run_temporal_remainder as loading
    engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=root
    try:
      with (REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        while True:
            stable()
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:state['status']='waiting_shared_lock';save();time.sleep(5)
        idle=0
        while idle<12:
            stable();foreign=engine.base.guard_records(GPU)
            if foreign:idle=0;state.update(status='waiting_foreign_process',foreign=foreign)
            else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
            save();time.sleep(30 if foreign else 10)
        control=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,timeout=5,
            env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=engine.PIPE))
        write(root/'private_control_before.json',dict(returncode=control.returncode,stdout=control.stdout,stderr=control.stderr))
        if control.returncode==0:raise ValueError('Do not reuse private MPS daemon')
        engine.daemon(True);active=True
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for key,task,old,mode in templates():
          for repeat in (1,2,3):
            stable();base=REPO/'LightGenV2/tasks'/task/'runs/simulation';template=base/old/'qwen_repeat01/identity.json'
            folder=base/('a100_first_tf_core_'+key+'_20261025_01')/f'qwen_repeat{repeat:02d}'
            if folder.exists():raise ValueError('Refuse to rerun existing point')
            name=f'{key}_qwen_repeat{repeat:02d}';deadline=min(args.deadline,time.time()+7199)
            python='/DATA/DATA1/guest3/t12_assets/venv/bin/python' if key=='08' else '/home/guest3/miniconda3/envs/xml/bin/python'
            cmd=[python,'-u',str(HERE/'first_tf_supplement_20261025.py'),'--template',str(template),'--output',str(folder),
                '--mode',mode,'--repeat',str(repeat),'--deadline',str(deadline)]
            write(root/(name+'.command.json'),cmd);observations=[]
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='running',phase=name,child_pid=child.pid,point=str(folder));save()
                try:
                    while child.poll() is None:
                        stable()
                        if time.time()>deadline:raise TimeoutError('Single worker deadline')
                        snapshot=loading.process_snapshot(GPU,child.pid,engine.PIPE)
                        if not (folder/'admission_bound.json').exists():observations.append(snapshot)
                        if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                            write(root/(name+'.foreign_interruption.json'),snapshot);raise RuntimeError('Foreign A100: stop own worker only')
                        time.sleep(1)
                except BaseException:
                    if child.poll() is None:
                        loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'first_tf_supplement_20261025.py');child.wait(timeout=40)
                    raise
                finally:write(root/(name+'.loading_process_audit.json'),observations)
            if child.returncode:raise RuntimeError(name+' failed; preserve raw, no blind retry')
            summary=json.loads((folder/'first_tf_summary.json').read_text());assert summary['status']=='complete' and summary['calls']==200
            state['completed'].append(dict(name=name,folder=str(folder),summary_sha256=sha(folder/'first_tf_summary.json')));save()
        state.update(status='complete',finished_unix=time.time())
    except BaseException as e:state.update(status='stopped',error=repr(e));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as e:state['mps_release_error']=repr(e)
        save()
if __name__=='__main__':main()
