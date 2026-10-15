// Node runtime and artifact-tool must come from Codex's bundled dependencies.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL, fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';

const output = path.dirname(fileURLToPath(import.meta.url));
const repository = path.resolve(output, '../../../../..');
const dependencyLoader = createRequire(path.join(repository, '.codex_tmp', 'artifact_loader.cjs'));
const {Workbook, SpreadsheetFile} = await import(pathToFileURL(dependencyLoader.resolve('@oai/artifact-tool')).href);
const input = JSON.parse(await fs.readFile(path.join(output, 'plot_data.json'), 'utf8'));
assert.equal(input.rows.length, 40);
const workbook = Workbook.create();
const figure1 = workbook.worksheets.add('Figure1');
const figure2 = workbook.worksheets.add('Figure2');
const sheet = workbook.worksheets.add('Comparison');
const protocol = workbook.worksheets.add('Protocol');
sheet.showGridLines = false; protocol.showGridLines = false;
sheet.getRange('A1:G47').format.font = {name: 'Arial', size: 10, color: '#202b36'};
sheet.getRange('A2').values = [['OpenMoji: laboratory Ours and digital Qwen']];
sheet.getRange('A2').format.font = {name: 'Arial', size: 14, bold: true};
sheet.getRange('A3').values = [['1,000 TEST scenes; 250 per operation; single-run development results']];
sheet.getRange('A5:G5').values = [['Group', 'n', 'Metric', 'Ours lab', 'Qwen digital', 'Delta (pp)', 'Relative gain (%)']];
const matrix = input.rows.map(r => [r.group, r.samples, r.label, r.ours, r.qwen, null, null]);
sheet.getRange('A6:G45').values = matrix;
for(let row=6; row<=45; row++){
  sheet.getRange(`F${row}:G${row}`).formulas = [[`=100*(D${row}-E${row})`, `=100*(D${row}/E${row}-1)`]];
}
sheet.getRange('A5:G5').format = {fill:'#23445b',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},horizontalAlignment:'center'};
sheet.getRange('A6:A45').format.columnWidth = 14;
sheet.getRange('B6:B45').format.columnWidth = 8;
sheet.getRange('C6:C45').format.columnWidth = 29;
sheet.getRange('D6:E45').format.columnWidth = 17;
sheet.getRange('F6:G45').format.columnWidth = 19;
sheet.getRange('D6:E45').setNumberFormat('0.00%');
sheet.getRange('F6:G45').setNumberFormat('+0.00;-0.00;0.00');
sheet.getRange('B6:B45').setNumberFormat('0');
sheet.getRange('A5:G45').format.rowHeight = 22;
sheet.getRange('B6:B45').format.horizontalAlignment = 'right';
sheet.getRange('D6:G45').format.horizontalAlignment = 'right';
for(const row of [6,14,22,30,38]){
  sheet.getRange(`A${row}:G${row}`).format.borders = {top:{style:'thin',color:'#c9d4dc'}};
}
sheet.freezePanes.freezeRows(5);
protocol.getRange('A1:B20').format.font = {name:'Arial',size:10,color:'#202b36'};
protocol.getRange('A2').values = [['Source identity and comparison limits']];
protocol.getRange('A2').format.font = {name:'Arial',size:14,bold:true};
const details = [
 ['Item','Value'],
 ['Ours report', 'ours_lab_full_report.json; complete, 1,000 TEST, 6,000 CCD captures'],
 ['Ours report SHA256', input.sources.files['ours_lab_full_report.json'].sha256],
 ['Ours PT SHA256', input.ours.checkpoint_sha256],
 ['Ours treatment', 'Frozen epoch45; no downstream retraining; no per-frame intensity normalization'],
 ['Qwen report', 'qwen_evaluation.json; selected epoch30; 1,000 TEST'],
 ['Qwen report SHA256', input.sources.files['qwen_evaluation.json'].sha256],
 ['Qwen PT SHA256', input.sources.qwen_checkpoint_sha256],
 ['Qwen backbone', 'Full frozen native Qwen3-VL-2B-Instruct features; trained adapters and shared readout'],
 ['Units', 'D/E are fractions formatted as percentages; F is percentage points; G is relative percent'],
 ['Test selection', input.protocol.test_role],
 ['Pairing limit', input.protocol.pairing_limit],
 ['Uncertainty', 'One sealed run; lab per-sample predictions unavailable here; no error bars or significance tests'],
 ['Baseline variant', input.protocol.baseline_variant],
];
protocol.getRange(`A5:B${4+details.length}`).values = details;
protocol.getRange('A5:B5').format = {fill:'#23445b',font:{name:'Arial',size:10,bold:true,color:'#ffffff'}};
protocol.getRange('A5:A18').format.columnWidth = 25;
protocol.getRange('B5:B18').format.columnWidth = 103;
protocol.getRange('A5:B18').format.rowHeight = 30;
protocol.getRange('B6:B18').format.wrapText = true;
// Two requested figure-order tables link back to the unchanged 40-row source.
assert.equal(input.figure1_rows.length, 8);
assert.equal(input.figure2_rows.length, 8);
for(const target of [figure1, figure2]){
  target.showGridLines = false;
  target.getRange('A1:H16').format.font = {name:'Arial',size:10,color:'#202b36'};
  target.getRange('A2').format.font = {name:'Arial',size:14,bold:true};
  target.getRange('A5:H5').format = {fill:'#23445b',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},horizontalAlignment:'center'};
  target.getRange('A5:H13').format.rowHeight = 24;
  target.getRange('A6:H13').format.verticalAlignment = 'center';
}
figure1.getRange('A2').values = [['Figure 1: overall semantic editing metrics']];
figure1.getRange('A3').values = [['LightGen-II is Ours laboratory; 1,000 TEST scenes per system']];
figure1.getRange('A5:H5').values = [['Order','Metric','Ours fraction','Qwen fraction','Ours (%)','Qwen (%)','Delta (pp)','Suggested use']];
figure1.getRange('A6:H13').values = input.figure1_rows.map(r=>[
  r.order,r.figure_label,null,null,null,null,null,r.recommended_for_main?'Main figure':'Supplement']);
figure2.getRange('A2').values = [['Figure 2: per-operation accuracy and edit IoU']];
figure2.getRange('A3').values = [['Panel a: changed-cell accuracy; panel b: edit-grid IoU; 250 scenes per operation']];
figure2.getRange('A5:H5').values = [['Panel','Operation','Metric','Ours fraction','Qwen fraction','Ours (%)','Qwen (%)','Delta (pp)']];
figure2.getRange('A6:H13').values = input.figure2_rows.map(r=>[
  r.panel,r.operation,r.label,null,null,null,null,null]);
for(let i=0;i<8;i++){
  const row=i+6, first=input.figure1_rows[i], second=input.figure2_rows[i];
  figure1.getRange(`C${row}:G${row}`).formulas = [[`=Comparison!D${first.source_row}`,`=Comparison!E${first.source_row}`,`=100*C${row}`,`=100*D${row}`,`=100*(C${row}-D${row})`]];
  figure2.getRange(`D${row}:H${row}`).formulas = [[`=Comparison!D${second.source_row}`,`=Comparison!E${second.source_row}`,`=100*D${row}`,`=100*E${row}`,`=100*(D${row}-E${row})`]];
}
figure1.getRange('A5:A13').format.columnWidth = 7;
figure1.getRange('B5:B13').format.columnWidth = 31;
figure1.getRange('C5:G13').format.columnWidth = 16;
figure1.getRange('H5:H13').format.columnWidth = 18;
figure2.getRange('A5:A13').format.columnWidth = 7;
figure2.getRange('B5:B13').format.columnWidth = 13;
figure2.getRange('C5:C13').format.columnWidth = 29;
figure2.getRange('D5:H13').format.columnWidth = 16;
figure1.getRange('C6:D13').setNumberFormat('0.000000');
figure2.getRange('D6:E13').setNumberFormat('0.000000');
figure1.getRange('E6:F13').setNumberFormat('0.00');
figure2.getRange('F6:G13').setNumberFormat('0.00');
figure1.getRange('G6:G13').setNumberFormat('+0.00;-0.00;0.00');
figure2.getRange('H6:H13').setNumberFormat('+0.00;-0.00;0.00');
figure1.getRange('C6:G13').format.horizontalAlignment = 'right';
figure2.getRange('D6:H13').format.horizontalAlignment = 'right';
figure1.getRange('A15').values = [['Fractions are 0-1; percentage columns are 0-100; delta is percentage points.']];
figure2.getRange('A15').values = [['Fractions are 0-1; percentage columns are 0-100; delta is percentage points.']];
workbook.recalculate();
const checked = sheet.getRange('D6:G45').values;
for(let i=0;i<input.rows.length;i++){
  assert(Math.abs(checked[i][0]-input.rows[i].ours)<1e-12);
  assert(Math.abs(checked[i][1]-input.rows[i].qwen)<1e-12);
  assert(Math.abs(checked[i][2]-input.rows[i].difference_pp)<1e-9);
  assert(Math.abs(checked[i][3]-input.rows[i].relative_change_percent)<1e-9);
}
const firstValues=figure1.getRange('C6:G13').values;
const secondValues=figure2.getRange('D6:H13').values;
for(const [values,rows] of [[firstValues,input.figure1_rows],[secondValues,input.figure2_rows]]){
  for(let i=0;i<rows.length;i++){
    const expected=[rows[i].ours,rows[i].qwen,100*rows[i].ours,100*rows[i].qwen,rows[i].difference_pp];
    for(let j=0;j<5;j++) assert(Math.abs(values[i][j]-expected[j])<1e-9);
  }
}
const errors = await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'formula errors'});
await fs.mkdir(path.join(output,'qa'),{recursive:true});
await fs.writeFile(path.join(output,'qa','workbook_error_scan.ndjson'),errors.ndjson);
for(const [name,range] of [['Figure1','A1:H15'],['Figure2','A1:H15'],['Comparison','A1:G15'],['Protocol','A1:B18']]){
  const preview = await workbook.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(output,'qa',`${name.toLowerCase()}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(output,'metrics.xlsx'));
// Machine-readable plotting data use the same verified workbook values, not rounded labels.
const csv = [['group','samples','metric','ours_fraction','qwen_fraction','difference_pp','relative_change_percent'],
  ...input.rows.map((r,i)=>[r.group,r.samples,r.metric,...checked[i]])];
await fs.writeFile(path.join(output,'plot_data.csv'),'\ufeff'+csv.map(row=>row.join(',')).join('\r\n')+'\r\n','utf8');
const quote = value => '"'+String(value).replaceAll('"','""')+'"';
const writeCsv = (name,rows) => fs.writeFile(path.join(output,name),'\ufeff'+rows.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n','utf8');
await writeCsv('figure1_data.csv',[
  ['order','metric','label','ours_fraction','qwen_fraction','ours_percent','qwen_percent','difference_pp','recommended_for_main'],
  ...input.figure1_rows.map((r,i)=>[r.order,r.metric,r.figure_label,...firstValues[i],r.recommended_for_main])]);
await writeCsv('figure2_data.csv',[
  ['panel','operation','metric','ours_fraction','qwen_fraction','ours_percent','qwen_percent','difference_pp'],
  ...input.figure2_rows.map((r,i)=>[r.panel,r.operation,r.metric,...secondValues[i]])]);
await fs.writeFile(path.join(output,'qa','workbook_audit.json'),JSON.stringify({rows:40,figure1_rows:8,figure2_rows:8,formulas_verified:160,all_report_values_preserved:true,cca_difference_pp:checked[0][2]},null,2)+'\n');
console.log(JSON.stringify({xlsx:path.join(output,'metrics.xlsx'),csv:path.join(output,'plot_data.csv'),rows:40,cca_difference_pp:checked[0][2]}));
