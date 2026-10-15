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
workbook.recalculate();
const checked = sheet.getRange('D6:G45').values;
for(let i=0;i<input.rows.length;i++){
  assert(Math.abs(checked[i][0]-input.rows[i].ours)<1e-12);
  assert(Math.abs(checked[i][1]-input.rows[i].qwen)<1e-12);
  assert(Math.abs(checked[i][2]-input.rows[i].difference_pp)<1e-9);
  assert(Math.abs(checked[i][3]-input.rows[i].relative_change_percent)<1e-9);
}
const errors = await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'formula errors'});
await fs.mkdir(path.join(output,'qa'),{recursive:true});
await fs.writeFile(path.join(output,'qa','workbook_error_scan.ndjson'),errors.ndjson);
for(const [name,range] of [['Comparison','A1:G15'],['Protocol','A1:B18']]){
  const preview = await workbook.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(output,'qa',`${name.toLowerCase()}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(output,'metrics.xlsx'));
// Machine-readable plotting data use the same verified workbook values, not rounded labels.
const csv = [['group','samples','metric','ours_fraction','qwen_fraction','difference_pp','relative_change_percent'],
  ...input.rows.map((r,i)=>[r.group,r.samples,r.metric,...checked[i]])];
await fs.writeFile(path.join(output,'plot_data.csv'),'\ufeff'+csv.map(row=>row.join(',')).join('\r\n')+'\r\n','utf8');
await fs.writeFile(path.join(output,'qa','workbook_audit.json'),JSON.stringify({rows:40,formulas_verified:80,all_report_values_preserved:true,cca_difference_pp:checked[0][2]},null,2)+'\n');
console.log(JSON.stringify({xlsx:path.join(output,'metrics.xlsx'),csv:path.join(output,'plot_data.csv'),rows:40,cca_difference_pp:checked[0][2]}));
