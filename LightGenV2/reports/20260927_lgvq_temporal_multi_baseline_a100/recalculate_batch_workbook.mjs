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
