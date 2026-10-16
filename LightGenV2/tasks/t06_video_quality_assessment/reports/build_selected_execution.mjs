import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const bundledRequire=createRequire(path.join(process.env.CODEX_ARTIFACT_RUNTIME_BASE??path.join(process.cwd(),'.codex_tmp'),'artifact-runtime.cjs'));
const {FileBlob,SpreadsheetFile}=await import(pathToFileURL(bundledRequire.resolve('@oai/artifact-tool')).href);
const repo=process.cwd();
const out=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013');
const run=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_selected_execution_a100_20261016');
const scratch=path.join(repo,'.codex_tmp');
function chooseDisplayPoints(points){
 const selected=new Set();
 for(const model of new Set(points.map(x=>x.baseline))){
  const values=points.filter(x=>x.baseline===model),higher=values.filter(x=>x.batch_size>1),near=higher.filter(x=>x.independent_resident_util_mean_percent>=95).slice(-2);
  for(const x of values.filter(x=>x.batch_size===1).concat(near.length?near:higher.slice(-1)))selected.add(model+':'+x.batch_size);
 }
 return selected;
}
if(process.argv.includes('--test-selection')){
 const cases=[['yolo',1,44],['yolo',2,73],['yolo',3,98],['clip',1,39],['clip',9,89],['clip',10,99],['qwen',3,97.1],['qwen',4,97.0],['fallback',1,50],['fallback',2,70],['fallback',3,80]];
 const actual=[...chooseDisplayPoints(cases.map(([baseline,batch_size,independent_resident_util_mean_percent])=>({baseline,batch_size,independent_resident_util_mean_percent})))];
 if(JSON.stringify(actual)!==JSON.stringify(['yolo:1','yolo:3','clip:1','clip:10','qwen:3','qwen:4','fallback:1','fallback:3']))throw Error('Display selection contract failed');
 console.log(JSON.stringify({selection_test_passed:true,midpoints_excluded:true,unmet_target_fallback:true}));process.exit(0);
}
const filenames=['plot_data.xlsx','LGVQ_temporal_multi_baseline_recalculated.xlsx','LGVQ_temporal_gpu_utilization.xlsx'];
const books=await Promise.all(filenames.map(async f=>SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out,f)))));
const [plot,main,util]=books;
if(process.argv.includes('--inspect')){
 for(let i=0;i<books.length;i++)console.log(filenames[i],(await books[i].inspect({kind:'sheet',include:'id,name',maxChars:1500})).ndjson);
 for(const [book,sheetName,range,file] of [[plot,'Comparison','A1:L17','plot'],[main,'Summary','A1:L13','summary'],[util,'GPU utilization','A1:I13','util']]){
  console.log((await book.inspect({kind:'table',range:sheetName+'!'+range,tableMaxRows:8,tableMaxCols:12,maxChars:2400})).ndjson);
  const image=await book.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(scratch,'selected_before_'+file+'.png'),new Uint8Array(await image.arrayBuffer()));
 }
 for(const [book,range]of [[plot,'Measurements!A1:T7'],[main,"'Batch results'!A1:AR2"],[main,'Sources!A1:B32'],[util,"'Repeat measurements'!A1:O6"]])console.log((await book.inspect({kind:'table',range,tableMaxRows:6,tableMaxCols:44,maxChars:4000})).ndjson);
 process.exit(0);
}
const json=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const data=await json(path.join(run,'verified_selected_execution.json'));
if(data.status!=='complete'||data.paired_rows.length!==16||data.repeats.length!==48)throw Error('All selected evidence must be complete and independently verified');
const palette={ours:['#7828A8','#F2E9FA'],qwen:['#0072B2','#EAF3FA'],glm:['#92527F','#F5EBF3'],small:['#C25B18','#FFF0E5'],clip:['#007F63','#E8F5EF'],yolo:['#9B7000','#FFF7DD'],deepseek:['#5B6470','#EEF0F3']};
const asNumber=v=>v===''||v===undefined||v===null?null:Number(v);
const anchor=data.historical_qwen_B1;
if(+anchor.time_16_videos_ms!==1188.3861464262009||+anchor.energy_16_videos_j!==462.1134010767563||+anchor.display_gpu_util_mean_percent!==63.133111480865225)throw Error('Historical anchor drift');
const checkpoint={qwen:'e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2'};
const oldMeasurements=plot.worksheets.getItem('Measurements');
const oldOursSource=oldMeasurements.getRange('S5').values[0][0];
const historicalFile=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/unified_qwen_b1_5s30s01/report.json');
const historical=await json(historicalFile),h=historical.measurement;
const rows=[{
 model:'LightGen-II',baseline:'ours',batch_size_videos:null,videos_per_optical_field:16,workload_videos:16,model_total_parameters_billion:null,moe_active_parameters_billion:null,
 measured_full558_srcc:data.ours.physical_reference_srcc,display_gpu_util_mean_percent:null,time_16_videos_ms:data.ours.time_16_videos_ms,energy_16_videos_j:data.ours.energy_16_videos_j,
 lightgen_time_speedup_x:1,lightgen_energy_efficiency_gain_x:1,same_model_time_gain_vs_b1_x:null,same_model_energy_gain_vs_b1_x:null,independent_resident_util_mean_percent:null,same_round_active_util_mean_percent:null,
 utilization_basis:'Not applicable: N=16 optical field',utilization_source:'',parameter_basis:'Optical degrees of freedom are not stored GPU model weights',parameter_source:'',cost_basis:'Unchanged physical reference and component-cost estimate',execution_profile:'physical optical system',cost_source:oldOursSource,checkpoint_sha256:'',cost_measured_local:'historical physical reference',util_measured_local:''
},{...anchor,batch_size_videos:1,videos_per_optical_field:null,workload_videos:16,model_total_parameters_billion:2,moe_active_parameters_billion:null,
 measured_full558_srcc:+anchor.measured_full558_srcc,display_gpu_util_mean_percent:+anchor.display_gpu_util_mean_percent,time_16_videos_ms:+anchor.time_16_videos_ms,energy_16_videos_j:+anchor.energy_16_videos_j,
 lightgen_time_speedup_x:+anchor.lightgen_time_speedup_x,lightgen_energy_efficiency_gain_x:+anchor.lightgen_energy_efficiency_gain_x,same_model_time_gain_vs_b1_x:1,same_model_energy_gain_vs_b1_x:1,
 independent_resident_util_mean_percent:+anchor.independent_resident_util_mean_percent,same_round_active_util_mean_percent:null,execution_profile:'historical native Qwen B1',cost_basis:'Protected historical anchor, not remeasured in this run',
 utilization_source:path.relative(repo,historicalFile).replaceAll('\\','/'),cost_source:oldMeasurements.getRange('S6').values[0][0],checkpoint_sha256:checkpoint.qwen,cost_measured_local:'historical finalized anchor',util_measured_local:'2026-10-15 14:52:05'}];
const b1=Object.fromEntries(data.paired_rows.filter(x=>x.batch_size===1).map(x=>[x.baseline,x]));
b1.qwen={time_16_videos_ms:+anchor.time_16_videos_ms,energy_16_videos_j:+anchor.energy_16_videos_j,srcc:+anchor.measured_full558_srcc};
const selectedKeys=chooseDisplayPoints(data.paired_rows);
for(const r of data.paired_rows){
 if(!selectedKeys.has(r.baseline+':'+r.batch_size))continue;
 const meta=r.parameter_metadata;
 rows.push({model:r.model,baseline:r.baseline,batch_size_videos:r.batch_size,videos_per_optical_field:null,workload_videos:16,
  model_total_parameters_billion:asNumber(meta.model_total_parameters_billion),moe_active_parameters_billion:asNumber(meta.moe_active_parameters_billion),
  measured_full558_srcc:r.srcc,display_gpu_util_mean_percent:r.independent_resident_util_mean_percent,time_16_videos_ms:r.time_16_videos_ms,energy_16_videos_j:r.energy_16_videos_j,
  lightgen_time_speedup_x:r.time_16_videos_ms/rows[0].time_16_videos_ms,lightgen_energy_efficiency_gain_x:r.energy_16_videos_j/rows[0].energy_16_videos_j,
  same_model_time_gain_vs_b1_x:b1[r.baseline].time_16_videos_ms/r.time_16_videos_ms,same_model_energy_gain_vs_b1_x:b1[r.baseline].energy_16_videos_j/r.energy_16_videos_j,
  independent_resident_util_mean_percent:r.independent_resident_util_mean_percent,same_round_active_util_mean_percent:r.gpu_util_mean_percent,
  utilization_basis:'Independent resident queue8, three complete 5s/30s repeats, all samples pooled',utilization_source:r.utilization_source,parameter_basis:meta.parameter_basis,parameter_source:meta.parameter_source,
  cost_basis:'Same-run first200 synchronized wall and active board-power estimate; '+r.execution,execution_profile:r.execution,cost_source:r.source,checkpoint_sha256:r.checkpoint_sha256,cost_measured_local:r.cost_measured_local,util_measured_local:r.measured_local});
}
for(const model of Object.keys(palette).filter(k=>k!=='ours')){const count=rows.filter(x=>x.baseline===model).length;if(count<2||count>3)throw Error('Need2–3 displayed points per GPU model');}
if(rows.length<13||rows.length>18)throw Error('Invalid sparse display');
for(const x of rows){const r=data.paired_rows.find(v=>v.baseline===x.baseline&&v.batch_size===x.batch_size_videos);Object.assign(x,{cost_wall_sum_ms:r?.cost_wall_sum_ms??null,cost_video_count:r?200:null,cost_call_count:r?.cost_calls??null,cost_active_A100_mean_W:r?.board_mean_w??null,cost_active_samples:r?.active_samples??null,util_repeat1_mean_percent:r?.util_repeat_means[0]??null,util_repeat2_mean_percent:r?.util_repeat_means[1]??null,util_repeat3_mean_percent:r?.util_repeat_means[2]??null,util_measurement_seconds:r?.util_duration_s??null,util_samples:r?.util_samples??null});}
const namesToArchive=[...filenames,'plot_data.csv','LGVQ_temporal_gpu_utilization.csv','LGVQ_temporal_gpu_utilization_repeats.csv','timing_energy_recalculated.csv','batch_metric_audit.csv','README.md','UTILIZATION_METHOD_20261015.md','latency_comparison_reference.png','latency_comparison_reference.svg','energy_comparison_reference.png','energy_comparison_reference.svg','srcc_cost_comparison_reference.png','srcc_cost_comparison_reference.svg','gpu_utilization_comparison.png','gpu_utilization_comparison.svg','batch_cost_comparison.png','batch_cost_comparison.svg'];
if(process.argv.includes('--verify')){
 const p=plot.worksheets.getItem('Comparison');
 for(let i=0;i<rows.length;i++){const r=i+7,x=rows[i];for(const[c,k]of[['E','measured_full558_srcc'],['G','time_16_videos_ms'],['H','energy_16_videos_j'],['I','lightgen_time_speedup_x'],['J','lightgen_energy_efficiency_gain_x']])if(Math.abs(p.getRange(c+r).values[0][0]-x[k])>1e-8)throw Error('Saved plot mismatch '+c+r);if(i&&Math.abs(p.getRange('F'+r).values[0][0]*100-x.display_gpu_util_mean_percent)>1e-8)throw Error('Saved util mismatch');}
 console.log(JSON.stringify({saved_rows_verified:rows.length,sparse_GPU_points:rows.length-1,Qwen_B1_preserved:true}));process.exit(0);
}
if(!process.argv.includes('--build'))throw Error('Use --inspect, --build or --verify');
const archive=path.join(run,'prior_report_archive');await fs.mkdir(archive,{recursive:true});const archived=[];
for(const name of namesToArchive){const input=path.join(out,name),bytes=await fs.readFile(input),target=path.join(archive,name);try{await fs.copyFile(input,target,fs.constants.COPYFILE_EXCL);}catch(error){if(error.code!=='EEXIST'||digest(await fs.readFile(target))!==digest(bytes))throw error;}archived.push({name,bytes:bytes.length,sha256:digest(bytes)});}
const archiveManifest=JSON.stringify({files:archived},null,2)+'\n';try{await fs.writeFile(path.join(archive,'manifest.json'),archiveManifest,{flag:'wx'});}catch(error){if(error.code!=='EEXIST'||await fs.readFile(path.join(archive,'manifest.json'),'utf8')!==archiveManifest)throw error;}
const comparison=plot.worksheets.getItem('Comparison'),inputs=plot.worksheets.getItem('Measurements'),summary=main.worksheets.getItem('Summary'),detail=main.worksheets.getItem('Batch results'),sources=main.worksheets.getItem('Sources'),uv=util.worksheets.getItem('GPU utilization'),ur=util.worksheets.getItem('Repeat measurements');
const na=v=>v===undefined||v===null||v===''?'n.a.':v;
const rawMap=new Map(data.paired_rows.map(x=>[x.baseline+':'+x.batch_size,x]));
const getRaw=x=>rawMap.get(x.baseline+':'+x.batch_size_videos);
comparison.getRange('A7:L44').unmerge();comparison.getRange('A7:L44').clear({applyTo:'all'});
comparison.getRange('A1').values=[['LGVQ 时间质量评价：精度与推理成本']];inputs.getRange('A1').values=[['Measurements and provenance']];
inputs.getRange('A5:T40').clear({applyTo:'all'});summary.getRange('A6:T43').clear({applyTo:'all'});detail.getRange('A2:AR38').clear({applyTo:'all'});
uv.getRange('A7:Q40').clear({applyTo:'all'});ur.getRange('A5:R100').clear({applyTo:'all'});
comparison.getRange('A2').values=[['A100 250 W。558 TEST 计算精度，首200视频成本归一到16视频。每模型仅展示 B1 及1–2个近满载点。']];
comparison.getRange('A3').values=[['Qwen B1 定稿不变。Qwen B3/4 为严格等价验证的优化执行，其余模型原生执行；不能把二者差异解释为纯 batch 效应。']];
comparison.getRange('A4').values=[['GPU 利用率为独立驻留测量，不代入成本功率。时延及能耗估计使用同轮交错推理数据。历史完整表和中间点已归档。']];
comparison.getRange('H6').values=[['16 视频\n能耗估计 (J)']];summary.getRange('E5').values=[['16-video estimated energy (J)']];detail.getRange('P1').values=[['16-video estimated energy (J)']];
inputs.getRange('A2').values=[['Columns F/K distinguish selected utilization from the separate historical B1 audit. Timing/power are paired; resident utilization is independent.']];
inputs.getRange('K4').values=[['Separate three-repeat resident audit (%)']];
summary.getRange('A3').values=[['Selected B1 and near-saturation points. Historical Qwen B1 retained; optimized Qwen B3/4 separately identified. Resident utilization is not cost telemetry.']];
summary.getRange('T5').values=[['Separate resident audit (%)']];detail.getRange('AQ1').values=[['Separate resident audit mean (%)']];uv.getRange('Q6').values=[['Separate resident audit mean (%)']];
ur.getRange('A3').values=[['Three complete 5s/30s queue8 repeats for each new selected point. Historical Qwen B1 retained separately. Resident power is not used for cost energy.']];
uv.getRange('A3').values=[['Selected B1 and near-saturation settings; historical Qwen B1 63.13% is retained, other rows are new three-repeat pooled means.']];
uv.getRange('A4').values=[['Qwen B3/4: equivalence-validated execution optimization. Other models: native. Same resident queue8 measurement method; no Graph or clock changes.']];
const inputValues=[],detailValues=[];
for(let i=0;i<rows.length;i++){
 const x=rows[i],raw=getRaw(x),retained=!raw,base=b1[x.baseline];
 inputValues.push([x.model,na(x.model_total_parameters_billion),na(x.moe_active_parameters_billion),x.baseline==='ours'?'N=16':x.batch_size_videos,x.measured_full558_srcc,na(x.display_gpu_util_mean_percent),x.time_16_videos_ms,x.energy_16_videos_j,na(base?.time_16_videos_ms),na(base?.energy_16_videos_j),na(x.independent_resident_util_mean_percent),na(x.same_round_active_util_mean_percent),x.utilization_basis,x.utilization_source,x.parameter_basis,x.parameter_source,x.cost_basis,x.checkpoint_sha256,x.cost_source,x.utilization_source]);
 detailValues.push([x.model,x.baseline==='ours'?16:x.batch_size_videos,x.baseline==='ours'?x.measured_full558_srcc:base.srcc,retained?'n.a.':200,na(raw?.cost_calls),na(raw?.wall_mean_ms),na(raw?.wall_p95_ms),na(raw?.board_mean_w),retained?'n.a.':338.2,x.baseline==='ours'?1:16/x.batch_size_videos,retained?1:0,retained?x.time_16_videos_ms:null,retained?x.energy_16_videos_j:null,null,null,null,null,null,null,x.cost_source,x.measured_full558_srcc,na(raw?.first200_srcc),x.execution_profile,'Original BF16/FP16 backbone and FP32 readout',x.energy_16_videos_j/16,'n.a.',na(raw?.cost_wall_sum_ms),na(raw?.gpu_util_mean_percent),na(raw?.gpu_util_peak_percent),na(raw?.memory_mean_mib),na(raw?.memory_peak_mib),na(raw?.active_samples),na(x.display_gpu_util_mean_percent),raw?raw.independent_resident_util_peak_percent:(x.baseline==='qwen'?h.peak_gpu_utilization_percent:'n.a.'),x.util_measured_local,x.utilization_source,x.cost_basis,retained?'n.a.':x.time_16_videos_ms,retained?'n.a.':x.energy_16_videos_j,na(raw?.plcc),na(x.model_total_parameters_billion),na(x.moe_active_parameters_billion),na(x.independent_resident_util_mean_percent),x.utilization_basis]);
}
inputs.getRange(`A5:T${rows.length+4}`).values=inputValues;detail.getRange(`A2:AR${rows.length+1}`).values=detailValues;
for(let i=0;i<rows.length;i++){
 const x=rows[i],r=i+7,ir=i+5,dr=i+2,sr=i+6,raw=getRaw(x),colors=palette[x.baseline];
 comparison.getRange(`A${r}:H${r}`).formulas=[[`=Measurements!A${ir}`,x.model_total_parameters_billion===null?'="n.a."':`=Measurements!B${ir}${x.model_total_parameters_billion<1?'*1000':''}`,x.moe_active_parameters_billion===null?'="n.a."':`=Measurements!C${ir}`,`=Measurements!D${ir}`,`=Measurements!E${ir}`,x.baseline==='ours'?'="n.a."':`=Measurements!F${ir}/100`,`=Measurements!G${ir}`,`=Measurements!H${ir}`]];
 comparison.getRange(`I${r}:L${r}`).formulas=[[`=G${r}/$G$7`,`=H${r}/$H$7`,x.baseline==='ours'?'="n.a."':`=Measurements!I${ir}/G${r}`,x.baseline==='ours'?'="n.a."':`=Measurements!J${ir}/H${r}`]];
 detail.getRange(`N${dr}:S${dr}`).formulas=[[raw?`=H${dr}+I${dr}`:'="n.a."',raw?`=AA${dr}*16/D${dr}`:`=L${dr}`,raw?`=N${dr}*O${dr}/1000`:`=M${dr}`,`=16/P${dr}`,`=O${dr}/$O$2`,`=P${dr}/$P$2`]];
 if(raw)detail.getRange(`F${dr}`).formulas=[[`=AA${dr}/E${dr}`]];
 detail.getRange(`Y${dr}`).formulas=[[`=P${dr}/16`]];
 summary.getRange(`A${sr}:T${sr}`).formulas=[[`='Batch results'!A${dr}`,x.baseline==='ours'?'="N=16"':`='Batch results'!B${dr}`,`='Batch results'!U${dr}`,`='Batch results'!O${dr}`,`='Batch results'!P${dr}`,`=16/E${sr}`,`=D${sr}/$D$6`,`=E${sr}/$E$6`,`='Batch results'!C${dr}`,`='Batch results'!V${dr}`,`='Batch results'!AG${dr}`,`='Batch results'!AH${dr}`,`='Batch results'!AB${dr}`,`='Batch results'!AE${dr}`,`='Batch results'!AK${dr}`,`='Batch results'!T${dr}`,`='Batch results'!AO${dr}`,`='Batch results'!AP${dr}`,`='Batch results'!AR${dr}`,`='Batch results'!AQ${dr}`]];
 for(const[sh,range]of[[comparison,`A${r}:L${r}`],[inputs,`A${ir}:T${ir}`],[summary,`A${sr}:T${sr}`],[detail,`A${dr}:AR${dr}`]]){sh.getRange(range).format={fill:colors[1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:27,verticalAlignment:'center'};}
 comparison.getRange(`A${r}`).format.font={name:'Arial',size:10,bold:true,color:colors[0]};comparison.getRange(`I${r}:J${r}`).format.font={name:'Arial',size:10,bold:true,color:colors[0]};
 comparison.getRange(`B${r}`).setNumberFormat(x.model_total_parameters_billion<1?'0.0" M"':'0.00" B"');comparison.getRange(`C${r}`).setNumberFormat('0.0" B"');
}
comparison.getRange(`E7:E${rows.length+6}`).setNumberFormat('0.0000');comparison.getRange(`F7:F${rows.length+6}`).setNumberFormat('0.00%');comparison.getRange(`G7:H${rows.length+6}`).setNumberFormat('#,##0.000');comparison.getRange(`I7:L${rows.length+6}`).setNumberFormat('0.00"×"');comparison.getRange(`B7:L${rows.length+6}`).format.horizontalAlignment='right';
for(const[sh,range]of[[summary,`C6:J${rows.length+5}`],[detail,`F2:S${rows.length+1}`]])sh.getRange(range).setNumberFormat('#,##0.000');
summary.getRange(`C6:C${rows.length+5}`).setNumberFormat('0.000000');summary.getRange(`K6:M${rows.length+5}`).setNumberFormat('0.00');
summary.getRange(`I6:J${rows.length+5}`).setNumberFormat('0.000000');detail.getRange(`U2:V${rows.length+1}`).setNumberFormat('0.000000');detail.getRange(`AN2:AN${rows.length+1}`).setNumberFormat('0.000000');detail.getRange(`C2:C${rows.length+1}`).setNumberFormat('0.000000');
const notes=['Qwen B1：历史原生定稿锚点，并非本轮复测。B3/4 使用经过完整558样本严格验证的执行优化。','LightGen-II 时间和能耗优势分别为 baseline T16 / LightGen-II T16、baseline E16 / LightGen-II E16。','同模型相对 B1：Qwen 同时包含 batch 与执行优化，其余模型为原生不同 batch 对照。','利用率为独立驻留负载的驱动窗口活跃比例，不是 SM occupancy，也不用于推算交错推理能耗。','本轮只冻结推理。既有 PT 若曾按 TEST 选模，相关精度仍属开发口径（无独立 VAL），并非新的独立泛化结论。'];
for(let i=0;i<notes.length;i++){const r=rows.length+8+i;comparison.getRange(`A${r}:L${r}`).merge();comparison.getRange(`A${r}`).values=[[notes[i]]];comparison.getRange(`A${r}:L${r}`).format={font:{name:'Microsoft YaHei',size:10,color:'#53687A'},rowHeight:25};}
const urows=rows.filter(x=>x.baseline!=='ours');
for(let i=0;i<urows.length;i++){
 const x=urows[i],raw=getRaw(x),r=i+7,means=raw?.util_repeat_means??[],samples=raw?.util_samples??h.samples,sum=raw?.util_sum_percent_points??(h.mean_gpu_utilization_percent*h.samples);
 uv.getRange(`A${r}:Q${r}`).values=[[x.model,x.batch_size_videos,null,raw?raw.independent_resident_util_peak_percent/100:h.peak_gpu_utilization_percent/100,raw?.util_duration_s??h.duration_s,samples,raw?.util_forward_calls??historical.measured_forward_invocations,raw?x.execution_profile:'Historical native, one retained run',new Date(x.util_measured_local.replace(' ','T')+'Z'),null,x.utilization_source,sum,na(means[0]),na(means[1]),na(means[2]),means.length?Math.sqrt(means.reduce((s,v)=>s+(v-means.reduce((a,b)=>a+b)/3)**2,0)/2):'n.a.',x.independent_resident_util_mean_percent]];
 uv.getRange(`C${r}`).formulas=[[`=L${r}/F${r}/100`]];uv.getRange(`A${r}:Q${r}`).format={fill:palette[x.baseline][1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:26};
}
uv.getRange(`C7:D${urows.length+6}`).setNumberFormat('0.00%');uv.getRange(`I7:I${urows.length+6}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
const historicRepeat={model:'Qwen3-VL-2B',baseline:'qwen',batch:1,repeat:1,mean_percent:h.mean_gpu_utilization_percent,peak_percent:h.peak_gpu_utilization_percent,duration_s:h.duration_s,samples:h.samples,forwards:historical.measured_forward_invocations,power_w:h.mean_board_power_w,memory_mib:h.mean_memory_used_mib,warmup_s:historical.warmup.duration_s,measured_local:'2026-10-15 14:52:05',source:path.relative(repo,historicalFile).replaceAll('\\','/'),role:'Historical native B1'};
const repeatRows=[historicRepeat,...data.repeats.map(x=>({...x,role:selectedKeys.has(x.baseline+':'+x.batch)?'New displayed point':'Audit only; not in main figure'}))];
ur.getRange(`A5:R${repeatRows.length+4}`).values=repeatRows.map(x=>[x.model,x.batch,x.repeat,x.mean_percent/100,x.peak_percent/100,x.duration_s,x.samples,x.forwards,x.power_w,x.memory_mib,na(x.sm_clock_mhz),na(x.temperature_c),na(x.cpu_queue_wait_ms),x.warmup_s,new Date(x.measured_local.replace(' ','T')+'Z'),x.baseline,x.role??'New three-repeat measurement',x.source]);
ur.getRange('P4:R4').values=[['Baseline key','Measurement identity','Raw source']];ur.getRange(`D5:E${repeatRows.length+4}`).setNumberFormat('0.00%');ur.getRange(`O5:O${repeatRows.length+4}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
for(let i=0;i<repeatRows.length;i++)ur.getRange(`A${i+5}:R${i+5}`).format={fill:palette[repeatRows[i].baseline][1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:25};
ur.getRange(`D5:E${repeatRows.length+4}`).setNumberFormat('0.00%');ur.getRange(`F5:N${repeatRows.length+4}`).setNumberFormat('0.00');ur.getRange(`G5:H${repeatRows.length+4}`).setNumberFormat('#,##0');ur.getRange(`O5:O${repeatRows.length+4}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
sources.getRange('A7:B7').values=[['Measured candidate batches','Qwen historical1 + optimized3/4; GLM1/2; Small1/3/4; CLIP1/9/10; YOLO1/2/3; Tiny1/6/8. Main view applies the95% rule below.']];
sources.getRange('A19:B20').values=[['Resident utilization','Independent unchanged queue8 three-repeat 5s/30s method; no Graph. Qwen B1 is one historical retained run.'],['Cost run','lgvq_selected_execution_a100_20261016;16 new sparse points, preserved historical Qwen B1']];
sources.getRange('A14:B14').values=[['Performance','Full558 evaluated at each batch. Previously TEST-selected checkpoints retain their development/no-VAL interpretation; no new training or selection.']];
sources.getRange('A23:B24').values=[['Dates','See each point source/driver telemetry for actual +0800 dates'],['Historical evidence','lgvq_selected_execution_a100_20261016/prior_report_archive/manifest.json']];
sources.getRange('A26:B31').values=[['Qwen execution','B3/B4: CPU integer metadata/indexing and grouped independent equal-length vision SDPA; original weights, precision and attention semantics.'],['Qwen equivalence','Full558 hidden/full vocabulary logits/quality logits/MOS compared exactly after cold cost in another process; each point has a receipt.'],['Gain definitions','Time/energy divided by LightGen-II reference. Qwen changes versus historical B1 include execution optimization and batch, not batch alone.'],['Model size','Total parameter definitions and original nominal/loaded sources retained in plot_data Measurements.'],['Selected data','B1 plus at most2 predeclared higher-B settings with resident mean>=95%. If none qualify, show highest predeclared B and disclose unmet target. Other measurements remain raw audit only.'],['Utilization and energy','Resident utilization is independent; energy uses same-cost-run active A100 power plus338.2W host proxy. Not a calibrated whole-system energy integral.']];
sources.getRange('A26:B31').format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true,rowHeight:50};
for(const book of books)book.recalculate();
const checkInput=inputs.getRange('G7'),oldInput=checkInput.values[0][0],oldGain=comparison.getRange('I9').values[0][0];
checkInput.values=[[oldInput*1.01]];plot.recalculate();if(Math.abs(comparison.getRange('G9').values[0][0]-oldInput*1.01)>1e-8||Math.abs(comparison.getRange('I9').values[0][0]-oldGain*1.01)>1e-8)throw Error('Plot formula recalculation failed');checkInput.values=[[oldInput]];plot.recalculate();
const checkWall=detail.getRange('AA4'),oldWall=checkWall.values[0][0],oldTime=summary.getRange('D8').values[0][0];checkWall.values=[[oldWall*1.01]];main.recalculate();if(Math.abs(summary.getRange('D8').values[0][0]-oldTime*1.01)>1e-8)throw Error('Main cost recalculation failed');checkWall.values=[[oldWall]];main.recalculate();
const checkCounter=uv.getRange('L8'),oldCounter=checkCounter.values[0][0],oldUtil=uv.getRange('C8').values[0][0];checkCounter.values=[[oldCounter*.99]];util.recalculate();if(Math.abs(uv.getRange('C8').values[0][0]-oldUtil*.99)>1e-10)throw Error('Utilization formula recalculation failed');checkCounter.values=[[oldCounter]];util.recalculate();
for(const [book,label,sheetName,range]of[[plot,'selected_plot','Comparison',`A1:L${rows.length+12}`],[plot,'selected_inputs','Measurements',`A1:L${rows.length+4}`],[main,'selected_summary','Summary',`A1:L${rows.length+5}`],[main,'selected_build','Batch results',`A1:S${rows.length+1}`],[main,'selected_sources','Sources','A19:B31'],[util,'selected_util','GPU utilization',`A1:I${urows.length+6}`],[util,'selected_repeats','Repeat measurements','A1:O18']]){const image=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(scratch,label+'.png'),new Uint8Array(await image.arrayBuffer()));}
for(const book of books)console.log((await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
for(let i=0;i<rows.length;i++){const r=i+7,x=rows[i];if(Math.abs(comparison.getRange('G'+r).values[0][0]-x.time_16_videos_ms)>1e-8||Math.abs(comparison.getRange('H'+r).values[0][0]-x.energy_16_videos_j)>1e-8||Math.abs(summary.getRange('D'+(i+6)).values[0][0]-x.time_16_videos_ms)>1e-8)throw Error('Formula/source mismatch');}
for(let i=0;i<books.length;i++)await(await SpreadsheetFile.exportXlsx(books[i])).save(path.join(out,filenames[i]));
const quote=v=>v===null||v===undefined?'':/[",\r\n]/.test(String(v))?'"'+String(v).replaceAll('"','""')+'"':String(v);
const csv=(objects,keys=[...new Set(objects.flatMap(x=>Object.keys(x)))])=>[keys,...objects.map(x=>keys.map(k=>x[k]))].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
await fs.writeFile(path.join(out,'plot_data.csv'),csv(rows));
await fs.writeFile(path.join(out,'timing_energy_recalculated.csv'),csv(rows));
await fs.writeFile(path.join(out,'batch_metric_audit.csv'),csv(data.paired_rows.map(x=>({model:x.model,batch_size_videos:x.batch_size,full558_srcc:x.srcc,full558_plcc:x.plcc,first200_srcc:x.first200_srcc,checkpoint_sha256:x.checkpoint_sha256,source:x.source,execution:x.execution}))));
await fs.writeFile(path.join(out,'LGVQ_temporal_gpu_utilization.csv'),csv(urows.map(x=>{const raw=getRaw(x);return{model:x.model,baseline:x.baseline,batch:x.batch_size_videos,mean_percent:x.display_gpu_util_mean_percent,peak_percent:raw?.independent_resident_util_peak_percent??h.peak_gpu_utilization_percent,repeat_means:raw?.util_repeat_means.join(',')??'',samples:raw?.util_samples??h.samples,duration_s:raw?.util_duration_s??h.duration_s,forwards:raw?.util_forward_calls??historical.measured_forward_invocations,execution:x.execution_profile,source:x.utilization_source,measured_local:x.util_measured_local};})));
await fs.writeFile(path.join(out,'LGVQ_temporal_gpu_utilization_repeats.csv'),csv(repeatRows));
await fs.writeFile(path.join(run,'presentation_selected.json'),JSON.stringify({status:'complete',rows,display_selection_rule:'B1 plus at most2 predeclared higher-B settings with resident mean>=95%; if none, show highest predeclared B as unmet target',source: 'verified_selected_execution.json',source_sha256:digest(await fs.readFile(path.join(run,'verified_selected_execution.json'))),prior_archive:'prior_report_archive/manifest.json',workbook_files:filenames},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({updated_workbooks:3,main_rows:rows.length,new_measurement_points:16,historical_Qwen_B1_unchanged:true}));
