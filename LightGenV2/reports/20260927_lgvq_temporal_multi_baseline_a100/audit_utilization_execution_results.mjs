// Read immutable completed controls, independently verify raw counters/calls.
// Never writes an XLSX, CSV, checkpoint, original report or metric.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const task=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment');
const run=path.join(task,'runs/simulation/lgvq_utilization_execution_20261015');
const suite=path.join(run,'gpu_suite02');
const reportDir=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const csv=p=>{
  const lines=fs.readFileSync(p,'utf8').trim().split(/\r?\n/);
  assert(!lines.some(x=>x.includes('"')),'This numeric telemetry parser rejects quoted fields');
  const header=lines.shift().split(',');
  return lines.map(line=>{
    const values=line.split(',');assert.equal(values.length,header.length);
    return Object.fromEntries(header.map((h,i)=>[h,values[i]]));
  });
};
const modes=['legacy_sync_cpu','sync_gpu_output','queued_gpu_output'];
const cases=['qwen_batch01','qwen_batch08','clip_batch08','clip_batch10','yolo_batch03','yolo_batch08','deepseek_batch08'];
const results=[], provenance={};
for(const name of cases){
  const folder=path.join(suite,name), p=path.join(folder,'report.json');
  if(!fs.existsSync(p))continue;
  const report=read(p);if(report.status!=='complete')continue;
  assert.deepEqual(report.modes.map(m=>m.mode),modes);
  assert.equal(report.gpu_uuid,'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55');
  assert.equal(report.source_sha256['audit_utilization_execution.py'],'a382ad8d1fea9db722cc3d08b16d22b306bfbd3e95d16dd95f6980fa9ec3fd6c');
  assert(report.historical_qwen_b1_anchor.unchanged);
  for(const mode of report.modes){
    const records=csv(path.join(folder,mode.mode+'_telemetry.csv'));
    const selected=records.filter(r=>r.phase===mode.mode);
    const rawU=selected.map(r=>Number(r.utilization_percent));
    const rawW=selected.map(r=>Number(r.watts));
    assert(rawU.length>300 && rawU.every(v=>Number.isFinite(v)&&v>=0&&v<=100));
    assert(rawW.every(v=>Number.isFinite(v)&&v>0));
    const calls=read(path.join(folder,mode.mode+'_calls.json'));
    assert.equal(calls.reduce((s,r)=>s+r.calls,0),mode.calls);
    assert(mode.duration_s>=30&&mode.duration_s<60);
    assert.equal(mode.windows[1].start_s,5);assert.equal(mode.windows[1].stop_s,30);
    assert(mode.last_mos.length===report.batch&&mode.last_mos.every(Number.isFinite));
    assert(mode.process_audit.every(r=>Number(r.pid)===report.pid));
    if(mode.mode!=='legacy_sync_cpu')assert(mode.max_abs_mos_vs_legacy<=.0002);
    const loopMs=mode.duration_s*16000/(mode.calls*report.batch);
    assert(Math.abs(loopMs-mode.complete_loop_time16_ms)<1e-8);
    const lastTime=Number(selected.at(-1).host_monotonic_s),firstTime=Number(selected[0].host_monotonic_s);
    results.push({case:name,mode:mode.mode,batch:report.batch,
      full_fixed_0_30s_utilization_percent:mode.windows[0].utilization_percent.mean,
      steady_fixed_5_30s_utilization_percent:mode.windows[1].utilization_percent.mean,
      raw_whole_phase_utilization_percent:mean(rawU),raw_whole_phase_mean_board_w:mean(rawW),
      raw_phase_samples:selected.length,raw_phase_recorded_span_s:lastTime-firstTime,
      complete_loop_time16_ms:loopMs,
      complete_loop_energy16_j:(338.2+mean(rawW))*loopMs/1000,
      first_block_sync_boundary_time16_ms:mode.boundary_wall_ms?mode.boundary_wall_ms.mean*16/report.batch:null,
      output_max_abs_delta:mode.max_abs_mos_vs_legacy??0,
      note:'Fixed-window U from source report. Entire-phase U and complete-loop costs independently recomputed. Queue full-forward throughput is not original first-block latency.'});
  }
  provenance[name]={report_sha256:sha(p),checkpoint_sha256:report.checkpoint_sha256,source_commit:report.source_commit};
}
let b1=null;
const first200=path.join(suite,'qwen_batch01/b1_new_audit_only.json');
if(fs.existsSync(first200)){
  const audit=read(first200),rows=csv(path.join(suite,'qwen_batch01/b1_first200_timing.csv'));
  assert.equal(rows.length,200);assert.deepEqual(rows.map(r=>Number(r.index)),Array.from({length:200},(_,i)=>i));
  const t=rows.reduce((s,r)=>s+Number(r.wall_ms),0)*16/200;
  const active=csv(path.join(suite,'qwen_batch01/b1_first200_telemetry.csv')).filter(r=>r.phase.startsWith('active:'));
  const w=mean(active.map(r=>Number(r.watts))),u=mean(active.map(r=>Number(r.utilization_percent)));
  assert(Math.abs(t-audit.cost.latency_16_videos_ms)<1e-8);
  assert(Math.abs((338.2+w)*t/1000-audit.cost.energy_16_videos_j)<1e-8);
  b1={historical_time16_ms:1188.3861464262009,new_audit_time16_ms:t,
      difference_percent:(t/1188.3861464262009-1)*100,new_audit_energy16_j:(338.2+w)*t/1000,
      new_audit_active_u_percent:u,historical_anchor_unchanged:true,historical_utilization_unavailable:true};
}
const priorManifest=read(path.join(task,'runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02/report_delivery_manifest.json'));
const protectedFiles=priorManifest.files.filter(f=>!f.name.endsWith('.png')&&!f.name.endsWith('.svg'));
for(const f of protectedFiles)assert.equal(sha(path.join(reportDir,f.name)),f.sha256,'Historical measurement/table changed: '+f.name);
if(process.argv.includes('--require-complete'))assert.equal(Object.keys(provenance).length,7);
const output={completed_cases:Object.keys(provenance).length,expected_cases:7,results,qwen_b1:b1,provenance,
  historical_data_files_verified_unchanged:protectedFiles.length,
  no_new_srcc_or_precision_changes:true,
  note:'Raw whole-phase and source fixed-window statistics are separate. Fixed windows were set before measurement, not chosen for maxima.'};
const target=path.join(run,'comparison_verified.json');
fs.writeFileSync(target,JSON.stringify(output,null,2)+'\n');
console.log(JSON.stringify({completed_cases:output.completed_cases,rows:results.length,qwen_b1:b1,historical_files_unchanged:protectedFiles.length}));
