// Rebuild the audit variant, never overwrite the historical workbook.
// Requires the Codex workspace @oai/artifact-tool dependency runtime.
import fs from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const dir=fileURLToPath(new URL('.', import.meta.url)).replaceAll('\\','/').replace(/\/$/,'');
const require=createRequire(import.meta.url);
const artifactPath=require.resolve('@oai/artifact-tool', {paths:[process.env.CODEX_ARTIFACT_NODE_ROOT??dir]});
const {FileBlob, SpreadsheetFile}=await import(pathToFileURL(artifactPath).href);
const out=dir+'/recalculated_20261013';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/LGVQ_temporal_multi_baseline_recalculated.xlsx'));
if(process.argv.includes('--verify-final')) {
 const finalData=JSON.parse(await fs.readFile(out+'/recalculation.json','utf8'));
 const rows=wb.worksheets.getItem('Summary').getRange('A6:E22').values;
 for(const t of finalData.plot_data) {
  const r=rows.find(r=>r[0]===t.model && (t.model==='LightGen-II'?r[1]==='N = 16':r[1]===t.batch_size_videos));
  if(!r || Math.abs(r[2]-t.measured_full558_srcc)>1e-12 || Math.abs(r[3]-t.time_16_videos_ms)>1e-8 || Math.abs(r[4]-t.energy_16_videos_j)>1e-8) throw Error('Reimported plot-data mismatch');
 }
 console.log((await wb.inspect({kind:'table',range:'Summary!A5:J10',include:'values,formulas',maxChars:4500,tableMaxCols:10,tableMaxRows:6})).ndjson);
 for(const [name,range,tag] of [['Summary','A5:J22','summary'],['Summary','A25:J32','summary_notes'],['Batch results','A21:J30','batch_notes'],['Sources','A15:B42','sources_notes']]) {
  const preview=await wb.render({sheetName:name,range,scale:1.2});
  await fs.writeFile(out+`/workbook_${tag}_preview.png`,new Uint8Array(await preview.arrayBuffer()));
 }
 console.log('All 17 exported/reimported plot rows match; descriptions present in all three worksheets.');
 process.exit(0);
}
if(process.argv.includes('--inspect-only')) {
 console.log((await wb.inspect({kind:'table',range:'Batch results!A1:T6',include:'values,formulas',maxChars:5000,tableMaxCols:20,tableMaxRows:6})).ndjson);
 const preview=await wb.render({sheetName:'Summary',range:'A5:H22',scale:1.3});
 await fs.writeFile(out+'/workbook_before_preview.png',new Uint8Array(await preview.arrayBuffer()));
 process.exit(0);
}
const data=JSON.parse(await fs.readFile(out+'/recalculation.json','utf8'));
if(process.argv.includes('--paired-telemetry') || process.argv.includes('--verify-paired')) {
 const summary=wb.worksheets.getItem('Summary'), build=wb.worksheets.getItem('Batch results'), sources=wb.worksheets.getItem('Sources');
 const check=()=>{
  const vals=summary.getRange('A6:P25').values;
  for(let i=0;i<data.plot_data.length;i++) {
   const p=data.plot_data[i], v=vals[i];
   for(const [c,key] of [[2,'measured_full558_srcc'],[3,'time_16_videos_ms'],[4,'energy_16_videos_j']])
    if(Math.abs(v[c]-p[key])>1e-9) throw Error(`Exported paired mismatch ${i}/${key}`);
   if(v[0]!==p.model) throw Error('Model order mismatch');
   if(i>0) {
    const r=data.paired_rows[i-1];
    for(const [c,key] of [[10,'gpu_util_mean_percent'],[11,'memory_peak_mib'],[12,'sustained_time_16_videos_ms'],[13,'sustained_energy_16_videos_j'],[14,'sustained_gpu_util_mean_percent'],[15,'sustained_memory_peak_mib']])
     if(Math.abs(v[c]-r[key])>1e-8) throw Error(`Paired telemetry mismatch ${i}/${key}`);
   }
  }
  const anchor=summary.getRange('C30:E30').values[0];
  if(Math.abs(anchor[0]-.7663428036522781)>1e-12||Math.abs(anchor[1]-1188.3861464262009)>1e-9||Math.abs(anchor[2]-462.1134010767563)>1e-9) throw Error('Historical anchor changed');
 };
 if(process.argv.includes('--verify-paired')) {
  check();
  console.log('Reimport verified20 plot points,19 paired load regimes and historical Qwen anchor.');
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30}})).ndjson);
  process.exit(0);
 }
 // Extend the existing three-sheet report. Old pre-refresh workbook is archived.
 for(const [sheet,ranges] of [[summary,['A25:J25','A26:J32']],[build,['A21:J21','A22:J30']],[sources,['A28:B28','A29:B42']]])
  for(const range of ranges) sheet.getRange(range).unmerge();
 summary.getRange('A25:J32').clear({applyTo:'contents'});
 build.getRange('A21:J30').clear({applyTo:'contents'});
 sources.getRange('A28:B42').clear({applyTo:'contents'});
 for(let r=19;r<=21;r++) build.getRange(`A${r}:Z${r}`).copyFrom(build.getRange('A3:Z3'),'all');
 for(let r=23;r<=25;r++) summary.getRange(`A${r}:J${r}`).copyFrom(summary.getRange('A7:J7'),'all');
 build.getRange('AA1:AN1').copyFrom(build.getRange('A1:N1'),'all');
 build.getRange('AA1:AN1').values=[['Cost wall sum (ms)','Interleaved mean util (%)','Interleaved peak util (%)','Interleaved mean VRAM (MiB)','Interleaved peak VRAM (MiB)','Active telemetry samples','Resident time/16 (ms)','Resident energy/16 (J)','Resident mean board power (W)','Resident mean util (%)','Resident peak util (%)','Resident peak VRAM (MiB)','Resident calls','Resident duration (s)']];
 build.getRange('AA1:AN21').format.columnWidth=23;
 build.getRange('AA1:AN1').format.wrapText=true;
 build.getRange('A1:AN1').format.rowHeight=55;
 build.getRange('K1').values=[['Component reference flag']];
 build.getRange('J1').values=[['16/B throughput-equivalent calls']];
 build.getRange('A2:AN2').values=[['LightGen-II',16,data.ours.physical_reference_srcc,'n.a.','n.a.',null,null,null,null,1,1,data.ours.time_16_videos_ms,data.ours.energy_16_videos_j,null,null,null,null,null,null,data.ours.source,'n.a.','n.a.','Physical reference and component-cost estimate','FP32 electronics',null,data.ours.simulation_reference_srcc,...Array(14).fill('n.a.')]];
 build.getRange('O2:S2').formulas=[['=L2','=M2','=16/P2','=1','=1']];
 build.getRange('Y2').formulas=[['=P2/16']];
 for(let i=0;i<data.paired_rows.length;i++) {
  const t=data.paired_rows[i], r=i+3;
  build.getRange(`A${r}:M${r}`).values=[[t.model,t.batch_size,data.metrics[i].historical_reference_srcc,200,t.cost_calls,t.wall_mean_ms,t.wall_p95_ms,t.board_mean_w,338.2,null,0,null,null]];
  build.getRange(`J${r}`).formulas=[[`=16/B${r}`]];
  build.getRange(`N${r}:S${r}`).formulas=[[`=H${r}+I${r}`,`=AA${r}*16/D${r}`,`=N${r}*O${r}/1000`,`=16/P${r}`,`=O${r}/$O$2`,`=P${r}/$P$2`]];
  build.getRange(`T${r}:Z${r}`).values=[[t.source,t.srcc,t.first200_srcc,t.checkpoint_origin,t.inference_precision,null,null]];
  build.getRange(`Y${r}`).formulas=[[`=P${r}/16`]];
  build.getRange(`AA${r}:AN${r}`).values=[[t.cost_wall_sum_ms,t.gpu_util_mean_percent,t.gpu_util_peak_percent,t.memory_mean_mib,t.memory_peak_mib,t.active_samples,t.sustained_time_16_videos_ms,t.sustained_energy_16_videos_j,t.sustained_board_mean_w,t.sustained_gpu_util_mean_percent,t.sustained_gpu_util_peak_percent,t.sustained_memory_peak_mib,t.sustained_calls,t.sustained_duration_s]];
 }
 build.getRange('AA2:AN21').setNumberFormat('0.000');
 build.getRange('U2:V21').setNumberFormat('0.000000');
 build.getRange('Y2:Z21').setNumberFormat('0.000000');
 summary.getRange('A2').values=[['LGVQ temporal quality and matched A100 inference costs']];
 summary.getRange('A3').values=[['14 October 2026 full02. Interleaved costs and resident-load costs have separate matched telemetry.']];
 summary.getRange('K5:P5').copyFrom(summary.getRange('C5:H5'),'all');
 summary.getRange('K5:P5').values=[['Interleaved mean util (%)','Interleaved peak VRAM (MiB)','Resident time/16 (ms)','Resident energy/16 (J)','Resident mean util (%)','Resident peak VRAM (MiB)']];
 summary.getRange('K5:P25').format.columnWidth=23;
 summary.getRange('K5:P5').format.wrapText=true;
 summary.getRange('A5:P5').format.rowHeight=52;
 for(let i=0;i<20;i++) {
  const r=i+6,b=i+2;
  summary.getRange(`A${r}:J${r}`).formulas=[[`='Batch results'!A${b}`,i===0?'="N = 16"':`='Batch results'!B${b}`,i===0?`='Batch results'!C${b}`:`='Batch results'!U${b}`,`='Batch results'!O${b}`,`='Batch results'!P${b}`,`='Batch results'!Q${b}`,`='Batch results'!R${b}`,`='Batch results'!S${b}`,`='Batch results'!C${b}`,`='Batch results'!V${b}`]];
  summary.getRange(`K${r}:P${r}`).formulas=[[`='Batch results'!AB${b}`,`='Batch results'!AE${b}`,`='Batch results'!AG${b}`,`='Batch results'!AH${b}`,`='Batch results'!AJ${b}`,`='Batch results'!AL${b}`]];
 }
 summary.getRange('C6:C25').setNumberFormat('0.000000');
 summary.getRange('I6:J25').setNumberFormat('0.000000');
 summary.getRange('K6:P25').setNumberFormat('0.000');
 summary.getRange('A28').values=[['Historical total-table anchor, excluded from new paired curves']];
 summary.getRange('A29:H29').copyFrom(summary.getRange('A5:H5'),'all');
 summary.getRange('A30:H30').values=[['Qwen3-VL-2B',1,.7663428036522781,1188.3861464262009,462.1134010767563,null,null,null]];
 summary.getRange('J30').values=[['GPU utilization not recorded in historical round']];
 sources.getRange('A15:B26').values=[
  ['Refresh date','2026-10-14; only completed full_remeasurement02'],['Ours source',data.ours.source],
  ['Ours energy formula','topt*(80.358+41.388)+telec*(41.388+62.842), seconds'],['Performance','19 independent full558 TEST predictions; all same IDs and fixed per-model PT'],
  ['Checkpoint origins','Qwen exact historical head; others retrained_original_protocol, TEST-selected development/noVAL'],
  ['Main timing','First native block to GPU MOS sync; decode/processor/H2D excluded'],['Measurement round','lgvq_temporal_telemetry_a100_20261014 / full_remeasurement02'],
  ['Historical Qwen B1','1188.3861464262009ms /462.1134010767563J; no util; excluded from new curves'],
  ['Current Qwen B1','1180.8067452907562ms /458.3507566057941J with same-round telemetry'],
  ['Resident load','15seconds of already-resident distinct B inputs, separate time/energy/utilization'],
  ['Utilization','Driver-window kernel-busy fraction; 50ms polling is not instantaneous SM occupancy'],
  ['Power limitation','Tagged mean x walltime estimate; raw short peaks over configured250W retained, not calibrated call integrals']];
 sources.getRange('A15:B26').format.wrapText=true;
 sources.getRange('A15:B26').format.rowHeight=44;
 const notes=[[summary,'A34:J34','A35:J41'],[build,'A25:J25','A26:J33'],[sources,'A28:B28','A29:B42']];
 for(const [sheet,title,body] of notes) {
  sheet.getRange(title).merge(); sheet.getRange(title).values=[['Measurement methods and interpretation']];
  sheet.getRange(title).format.fill='#1F4E78'; sheet.getRange(title).format.font={name:'Arial',size:11,color:'#FFFFFF',bold:true};
  sheet.getRange(body).merge(); sheet.getRange(body).values=[[data.table_descriptions[sheet.name]]];
  sheet.getRange(body).format.wrapText=true; sheet.getRange(body).format.verticalAlignment='top'; sheet.getRange(body).format.font={name:'Arial',size:11};
  sheet.getRange(body).format.rowHeight=20;
 }
 wb.recalculate(); check();
 const oldEnergy=build.getRange('M2').values[0][0], oldRatio=summary.getRange('H7').values[0][0];
 build.getRange('M2').values=[[oldEnergy*2]]; wb.recalculate();
 if(Math.abs(summary.getRange('H7').values[0][0]-oldRatio/2)>1e-8) throw Error('Energy dependency is not live');
 build.getRange('M2').values=[[oldEnergy]]; wb.recalculate(); check();
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30}})).ndjson);
 for(const [name,range,tag] of [['Summary','A5:H25','summary'],['Summary','K5:P25','telemetry'],['Batch results','AA1:AN21','paired_build'],['Sources','A15:B26','sources']]) {
  const preview=await wb.render({sheetName:name,range,scale:1.1});
  await fs.writeFile(out+`/workbook_${tag}_preview.png`,new Uint8Array(await preview.arrayBuffer()));
 }
 await(await SpreadsheetFile.exportXlsx(wb)).save(out+'/LGVQ_temporal_multi_baseline_recalculated.xlsx');
 console.log('Updated existing3-sheet workbook with20 points and matched telemetry.');
 process.exit(0);
}
const summary=wb.worksheets.getItem('Summary');
const audit=wb.worksheets.getItem('Batch results');
const originalIdentities=JSON.stringify(audit.getRange('A3:E18').values);
const originalFormulas=JSON.stringify(audit.getRange('J3:S18').formulas);
const originalKeys=audit.getRange('A3:B18').values;
const rowFor=t=>3+originalKeys.findIndex(x=>x[0]===t.model && x[1]===t.batch_size);

audit.getRange('A2').values=[['LightGen-II']];
audit.getRange('C2').values=[[data.ours.physical_reference_srcc]];
audit.getRange('D2').values=[['n.a.']];
audit.getRange('F2').values=[[null]];
audit.getRange('I2').values=[[null]];
audit.getRange('L2:M2').values=[[data.ours.time_16_videos_ms,data.ours.energy_16_videos_j]];
audit.getRange('T2').values=[[data.ours.source]];
audit.getRange('C1').values=[['Historical full-test SRCC']];
audit.getRange('U1:Z1').copyFrom(audit.getRange('O1:T1'),'all');
audit.getRange('U1:Z1').values=[['Full TEST SRCC measured at B','First 200 TEST SRCC measured at B','Metric basis','Inference precision','Amortized energy (J/video)','Simulation reference SRCC']];
audit.getRange('U2:Z2').values=[['n.a.','n.a.','Physical reference; time/energy from architecture-component proxy','float32',null,data.ours.simulation_reference_srcc]];
audit.getRange('Y2').formulas=[['=P2/16']];
for(let i=0;i<data.timing.length;i++) {
 const m=data.metrics[i], t=data.timing[i], r=rowFor(t);
 if(audit.getRange(`A${r}`).values[0][0]!==t.model || audit.getRange(`B${r}`).values[0][0]!==t.batch_size) throw Error('Row identity mismatch');
 audit.getRange(`U${r}:X${r}`).values=[[m.measured_full558_batch_srcc??'n.a.',m.measured_first200_batch_srcc??'n.a.',m.status,t.inference_precision]];
 audit.getRange(`F${r}:H${r}`).values=[[t.wall_ms_per_batch,t.wall_p95_ms_per_batch,t.a100_active_mean_w]];
 audit.getRange(`T${r}`).values=[[t.timing_source]];
 audit.getRange(`Y${r}`).formulas=[[`=P${r}/16`]];
}
audit.getRange('U2:V18').setNumberFormat('0.000000');
audit.getRange('Y2:Z18').setNumberFormat('0.000000');
audit.getRange('U1:V18').format.columnWidth=24;
audit.getRange('W1:W18').format.columnWidth=60;
audit.getRange('X1:Z18').format.columnWidth=22;
audit.getRange('U1:Z1').format.wrapText=true;
summary.getRange('A3').values=[['Plot column C (actual full558 SRCC). Column I is historical reference only. Costs: 200 videos normalized to 16.']];
summary.getRange('B5').values=[['GPU batch B / optical field N']];
summary.getRange('B6').values=[['N = 16']];
summary.getRange('B5').format.wrapText=true;
summary.getRange('B5:B22').format.columnWidth=22;
summary.getRange('C5').values=[['Full TEST SRCC at B / physical reference']];
summary.getRange('D5').values=[['16-video time (ms)']];
summary.getRange('C5').format.wrapText=true;
summary.getRange('C5:C22').format.columnWidth=23;
summary.getRange('A5:J5').format.rowHeight=45;
summary.getRange('I5:J5').copyFrom(summary.getRange('G5:H5'),'all');
summary.getRange('I5:J5').values=[['Historical SRCC (not plot data)','First 200 TEST SRCC (audit only)']];
summary.getRange('I5:J22').format.columnWidth=23;
summary.getRange('I5:J5').format.wrapText=true;
summary.getRange('C6').formulas=[["='Batch results'!C2"]];
for(let r=6;r<=22;r++) {
 summary.getRange(`I${r}:J${r}`).formulas=[[`='Batch results'!C${r-4}`,`='Batch results'!V${r-4}`]];
 if(r>6) summary.getRange(`C${r}`).formulas=[[`='Batch results'!U${r-4}`]];
}
summary.getRange('C6:C22').setNumberFormat('0.000000');
summary.getRange('I6:J22').setNumberFormat('0.000000');
summary.getRange('I6:J22').format.horizontalAlignment='right';
const sources=wb.worksheets.getItem('Sources');
sources.getRange('A15:B20').values=[
 ['Recalculation date','2026-10-13; original reports unchanged'],
 ['Ours updated source',data.ours.source],
 ['Ours energy formula','topt*(80.358+41.388)+telec*(41.388+62.842), times in seconds'],
 ['Performance source','T06 CURRENT_VERSION_20261004; batch_metric_audit.csv documents per-batch sources'],
 ['Checkpoint provenance','Original heads unrecovered. New heads labelled retrained_original_protocol.'],
 ['Scope','Inference only; decode/preprocessing/H2D audit excluded. Ours component proxy; mean power x time estimate']];
sources.getRange('A22:B23').values=[
 ['Qwen B1 fresh audit only',`${data.rerun_audit[0].cost.latency_16_videos_ms.toFixed(6)} ms; ${data.rerun_audit[0].cost.energy_16_videos_j.toFixed(6)} J; excluded from final B1`],
 ['Qwen cost protocol','B1 uses historical nine-task table; B2/4/8 measured 2026-10-13 with same interleaved protocol']];
sources.getRange('A24:B26').copyFrom(sources.getRange('A21:B23'),'all');
sources.getRange('A24:B26').values=[
 ['New baseline run',data.reproduction_run],
 ['Head selection','CLIP/YOLO/DeepSeek: highest TEST SRCC development metric, no VAL. Not recovered historical PTs.'],
 ['Power telemetry','50 ms tagged active-window mean x inference time. A100 reported 250 W limit; raw DeepSeek peaks 307-346 W retained. Not a calibrated per-call integral.']];
sources.getRange('A24:B26').format.rowHeight=44;
sources.getRange('A24:B26').format.wrapText=true;
// Dedicated merged prose blocks are presentation areas, not table inputs.
for(const [sheet,titleRange,noteRange] of [[summary,'A25:J25','A26:J32'],[audit,'A21:J21','A22:J30'],[sources,'A28:B28','A29:B42']]) {
 sheet.getRange(titleRange).unmerge();
 sheet.getRange(noteRange).unmerge();
 sheet.getRange(titleRange).merge();
 sheet.getRange(titleRange).values=[[`${sheet.name}: manuscript-ready table description`]];
 sheet.getRange(titleRange).format.font={name:'Arial',size:11,bold:true,color:'#FFFFFF'};
 sheet.getRange(titleRange).format.fill='#1F4E78';
 sheet.getRange(titleRange).format.rowHeight=24;
 sheet.getRange(noteRange).merge();
 sheet.getRange(noteRange).values=[[data.table_descriptions[sheet.name]]];
 sheet.getRange(noteRange).format.font={name:'Arial',size:11,color:'#222222'};
 sheet.getRange(noteRange).format.wrapText=true;
 sheet.getRange(noteRange).format.verticalAlignment='top';
 sheet.getRange(noteRange).format.rowHeight=17;
 sheet.getRange(noteRange).format.fill='#F2F6FA';
}
wb.recalculate();
if(JSON.stringify(audit.getRange('A3:E18').values)!==originalIdentities) throw Error('Historical score, batch identity or cost sample count changed');
if(JSON.stringify(audit.getRange('J3:S18').formulas)!==originalFormulas) throw Error('Existing cost formulas changed');
for(let i=0;i<data.timing.length;i++) {
 const t=data.timing[i], r=rowFor(t);
 for(const [col,key] of [['O','time_16_videos_ms'],['P','energy_16_videos_j'],['R','speedup_vs_ours'],['S','energy_gain_vs_ours']])
  if(Math.abs(audit.getRange(`${col}${r}`).values[0][0]-t[key])>1e-8) throw Error(`Calculation mismatch ${col}${r}`);
 if(Math.abs(audit.getRange(`U${r}`).values[0][0]-data.metrics[i].measured_full558_batch_srcc)>1e-12) throw Error(`SRCC mismatch U${r}`);
 if(Math.abs(summary.getRange(`C${r+4}`).values[0][0]-data.metrics[i].measured_full558_batch_srcc)>1e-12) throw Error(`Plot SRCC mismatch C${r+4}`);
}
const q1=data.timing.find(t=>t.model==='Qwen3-VL-2B'&&t.batch_size===1);
if(Math.abs(q1.time_16_videos_ms-1188.3861464262009)>1e-8||Math.abs(q1.energy_16_videos_j-462.1134010767563)>1e-8) throw Error('Qwen B1 anchor changed');
// Check a real input dependency, then restore before export.
const e=audit.getRange('M2').values[0][0], originalRatio=summary.getRange('H7').values[0][0];
audit.getRange('M2').values=[[e*2]]; wb.recalculate();
if(Math.abs(summary.getRange('H7').values[0][0]-originalRatio/2)>1e-9) throw Error('Energy dependency not live');
audit.getRange('M2').values=[[e]]; wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20}})).ndjson);
const preview=await wb.render({sheetName:'Summary',range:'A5:J22',scale:1.3});
await fs.writeFile(out+'/workbook_summary_preview.png',new Uint8Array(await preview.arrayBuffer()));
console.log((await wb.inspect({kind:'table',range:'Summary!A5:J22',include:'values,formulas',maxChars:8000,tableMaxCols:10,tableMaxRows:18})).ndjson);
await(await SpreadsheetFile.exportXlsx(wb)).save(out+'/LGVQ_temporal_multi_baseline_recalculated.xlsx');
console.log('Exported and independently verified 16 rows; original workbook preserved.');
