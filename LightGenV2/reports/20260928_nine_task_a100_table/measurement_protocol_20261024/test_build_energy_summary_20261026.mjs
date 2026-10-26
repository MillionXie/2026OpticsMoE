// Pure layout tests only: never call Workbook.create, export, render, or GPU.
import test from 'node:test';
import assert from 'node:assert/strict';
import {layout,col,stages} from './build_energy_summary_20261026.mjs';

function fixture(){
  const rounds=[],groups=[];
  for(const model of ['qwen','glm','clip','yolo'])for(const cap of [100,50]){
    for(const repeat of [1,2,3])rounds.push({platform:'A10040GB_PCIE_250W',model,cap,repeat});
    groups.push({platform:'A10040GB_PCIE_250W',model,cap,rounds:3});
  }
  for(const [model,n] of [['kimi',3],['deepseek',2]]){
    for(let repeat=1;repeat<=n;repeat++)rounds.push({platform:'A10080GB_SXM4_400W',model,cap:50,repeat});
    groups.push({platform:'A10080GB_SXM4_400W',model,cap:50,rounds:n});
  }
  return {status:'final_qualified_inputs',lab_complete_points:24,LG_async_measured:true,LG:{},missing_lab_points:[],rounds,groups};
}

test('all 29 available rounds mapped exactly once, unequal rental count preserved',()=>{
  const p=layout(fixture());assert.equal(p.rounds.length,29);assert.equal(p.groups.length,10);
  assert.equal(p.groups.find(x=>x.model==='deepseek').hi-p.groups.find(x=>x.model==='deepseek').lo+1,2);
  assert.equal(p.rounds[0].row,2);assert.equal(p.rounds.at(-1).row,30);assert.equal(p.lgInputStart,34);
});
test('group ranges are contiguous and their totals never include LG input rows',()=>{
  const p=layout(fixture());for(let i=1;i<p.groups.length;i++)assert.equal(p.groups[i].lo,p.groups[i-1].hi+1);
  assert.ok(p.groups.at(-1).hi<p.lgInputStart);
});
test('partial or missing LG cannot author final workbook',()=>{
  for(const edit of [d=>d.status='partial_not_final',d=>d.LG=null,d=>d.LG_async_measured=false,d=>d.lab_complete_points=23,d=>d.missing_lab_points.push('qwen')]){
    const d=fixture();edit(d);assert.throws(()=>layout(d),/qualified/);
  }
});
test('wrong cap and duplicate lab identity rejected',()=>{
  let d=fixture();d.rounds[0].cap=75;assert.throws(()=>layout(d),/identities/);
  d=fixture();d.rounds[0]={...d.rounds[1]};assert.throws(()=>layout(d),/identities/);
});
test('group count and unmapped original points cannot silently disappear',()=>{
  let d=fixture();d.groups[0].rounds=2;assert.throws(()=>layout(d),/count/);
  d=fixture();d.groups.pop();assert.throws(()=>layout(d),/Unmapped/);
});
test('column indices cover source hashes without off-by-one shifts',()=>{
  assert.equal(col(0),'A');assert.equal(col(13),'N');assert.equal(col(22),'W');assert.equal(col(23),'X');assert.equal(col(27),'AB');
  assert.equal(stages.length,10);assert.equal(stages[2][0],'CPU_tokenizer_retained_ms');
});

test('36 original rounds map to 12 non-overlapping three-round groups',()=>{
  const d=fixture();
  d.rounds=d.rounds.filter(p=>p.platform==='A10040GB_PCIE_250W');
  d.groups=d.groups.filter(p=>p.platform==='A10040GB_PCIE_250W');
  for(const model of ['kimi','deepseek'])for(const cap of [50,100]){
    d.groups.push({platform:'A10080GB_SXM4_400W',model,cap,rounds:3});
    for(const repeat of [1,2,3])d.rounds.push({platform:'A10080GB_SXM4_400W',model,cap,repeat});
  }
  const p=layout(d);assert.equal(p.rounds.length,36);assert.equal(p.groups.length,12);
  assert.equal(p.groups.at(-1).summaryRow,18);assert.equal(p.rounds.at(-1).row,37);
  assert.equal(p.lgInputStart,41);
});
