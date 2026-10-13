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
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(process.argv.includes('--verify-final')?out+'/LGVQ_temporal_multi_baseline_recalculated.xlsx':dir+'/LGVQ_temporal_multi_baseline_A100.xlsx'));
if(process.argv.includes('--verify-final')) {
 console.log((await wb.inspect({kind:'table',range:'Summary!A5:J10',include:'values,formulas',maxChars:4500,tableMaxCols:10,tableMaxRows:6})).ndjson);
 for(const [name,range] of [['Batch results','A1:J6'],['Sources','A15:B23']]) {
  const preview=await wb.render({sheetName:name,range,scale:1.2});
  await fs.writeFile(out+`/workbook_${name.replaceAll(' ','_')}_preview.png`,new Uint8Array(await preview.arrayBuffer()));
 }
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
const unchangedRows=Array.from({length:16},(_,i)=>i+3).filter(r=>audit.getRange(`A${r}`).values[0][0]!=='Qwen3-VL-2B');
const originalBaselines=JSON.stringify(unchangedRows.map(r=>audit.getRange(`A${r}:P${r}`).values));
const originalKeys=audit.getRange('A3:B18').values;
const rowFor=t=>3+originalKeys.findIndex(x=>x[0]===t.model && x[1]===t.batch_size);

audit.getRange('A2').values=[['LightGen-II']];
audit.getRange('C2').values=[[data.ours.physical_reference_srcc]];
audit.getRange('D2').values=[['n.a.']];
audit.getRange('F2').values=[[null]];
audit.getRange('I2').values=[[null]];
audit.getRange('L2:M2').values=[[data.ours.time_16_videos_ms,data.ours.energy_16_videos_j]];
audit.getRange('T2').values=[[data.ours.source]];
audit.getRange('C1').values=[['Full-test reference SRCC']];
audit.getRange('U1:Z1').copyFrom(audit.getRange('O1:T1'),'all');
audit.getRange('U1:Z1').values=[['Full TEST SRCC measured at B','First 200 TEST SRCC measured at B','Metric basis','Inference precision','Amortized energy (J/video)','Simulation reference SRCC']];
audit.getRange('U2:Z2').values=[['n.a.','n.a.','Physical reference; time/energy from architecture-component proxy','float32',null,data.ours.simulation_reference_srcc]];
audit.getRange('Y2').formulas=[['=P2/16']];
for(let i=0;i<data.timing.length;i++) {
 const m=data.metrics[i], t=data.timing[i], r=rowFor(t);
 if(audit.getRange(`A${r}`).values[0][0]!==t.model || audit.getRange(`B${r}`).values[0][0]!==t.batch_size) throw Error('Row identity mismatch');
 audit.getRange(`U${r}:X${r}`).values=[[m.measured_full558_batch_srcc??'n.a.',m.measured_first200_batch_srcc??'n.a.',m.status,t.inference_precision]];
 if(t.model==='Qwen3-VL-2B') {
  audit.getRange(`F${r}`).values=[[t.wall_ms_per_batch]];
  audit.getRange(`G${r}`).values=[[t.wall_p95_ms_per_batch]];
  audit.getRange(`H${r}`).values=[[t.a100_active_mean_w]];
  audit.getRange(`T${r}`).values=[[t.timing_source]];
 }
 audit.getRange(`Y${r}`).formulas=[[`=P${r}/16`]];
}
audit.getRange('U2:V18').setNumberFormat('0.000000');
audit.getRange('Y2:Z18').setNumberFormat('0.000000');
audit.getRange('U1:V18').format.columnWidth=24;
audit.getRange('W1:W18').format.columnWidth=60;
audit.getRange('X1:Z18').format.columnWidth=22;
audit.getRange('U1:Z1').format.wrapText=true;
summary.getRange('A3').values=[['Qwen: total-table B1 retained; B2/4/8 frozen reevaluation. Other models: historical costs and reference SRCC.']];
summary.getRange('C5').values=[['Full-test reference SRCC']];
summary.getRange('D5').values=[['16-video time (ms)']];
summary.getRange('C5').format.wrapText=true;
summary.getRange('C5:C22').format.columnWidth=23;
summary.getRange('A5:J5').format.rowHeight=34;
summary.getRange('I5:J5').copyFrom(summary.getRange('G5:H5'),'all');
summary.getRange('I5:J5').values=[['Full TEST SRCC at B','First 200 TEST SRCC at B']];
summary.getRange('I5:J22').format.columnWidth=23;
summary.getRange('I5:J5').format.wrapText=true;
for(let r=6;r<=22;r++) summary.getRange(`I${r}:J${r}`).formulas=[[`='Batch results'!U${r-4}`,`='Batch results'!V${r-4}`]];
summary.getRange('I6:J22').setNumberFormat('0.000000');
summary.getRange('I6:J22').format.horizontalAlignment='right';
const sources=wb.worksheets.getItem('Sources');
sources.getRange('A15:B20').values=[
 ['Recalculation date','2026-10-13; original reports unchanged'],
 ['Ours updated source',data.ours.source],
 ['Ours energy formula','topt*(80.358+41.388)+telec*(41.388+62.842), times in seconds'],
 ['Performance source','T06 CURRENT_VERSION_20261004; batch_metric_audit.csv documents per-batch sources'],
 ['Missing evaluation',data.missing],
 ['Scope','Inference only; decode/preprocessing/H2D audit excluded. Ours component proxy; mean power x time estimate']];
sources.getRange('A22:B23').values=[
 ['Qwen B1 fresh audit only',`${data.rerun_audit[0].cost.latency_16_videos_ms.toFixed(6)} ms; ${data.rerun_audit[0].cost.energy_16_videos_j.toFixed(6)} J; excluded from final B1`],
 ['Qwen cost protocol','B1 uses historical nine-task table; B2/4/8 measured 2026-10-13 with same interleaved protocol']];
wb.recalculate();
if(JSON.stringify(unchangedRows.map(r=>audit.getRange(`A${r}:P${r}`).values))!==originalBaselines) throw Error('Unrelated baseline raw inputs changed');
for(let i=0;i<data.timing.length;i++) {
 const t=data.timing[i], r=rowFor(t);
 for(const [col,key] of [['O','time_16_videos_ms'],['P','energy_16_videos_j'],['R','speedup_vs_ours'],['S','energy_gain_vs_ours']])
  if(Math.abs(audit.getRange(`${col}${r}`).values[0][0]-t[key])>1e-8) throw Error(`Calculation mismatch ${col}${r}`);
}
// Check a real input dependency, then restore before export.
const e=audit.getRange('M2').values[0][0], originalRatio=summary.getRange('H7').values[0][0];
audit.getRange('M2').values=[[e*2]]; wb.recalculate();
if(Math.abs(summary.getRange('H7').values[0][0]-originalRatio/2)>1e-9) throw Error('Energy dependency not live');
audit.getRange('M2').values=[[e]]; wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20}})).ndjson);
const preview=await wb.render({sheetName:'Summary',range:'A5:J22',scale:1.3});
await fs.writeFile(out+'/workbook_summary_preview.png',new Uint8Array(await preview.arrayBuffer()));
await(await SpreadsheetFile.exportXlsx(wb)).save(out+'/LGVQ_temporal_multi_baseline_recalculated.xlsx');
console.log('Exported and independently verified 16 rows; original workbook preserved.');
