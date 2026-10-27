// Fixed-value comparison plus a source-derived LightGen phase table.
import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const [input,out]=process.argv.slice(2);
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.status!=='raw_rechecked_fixed_values'||data.rows.length!==13||data.headers.length!==12)throw Error('Audited full fixed-value cohort required');
try{await fs.access(out);throw Error('Preserve existing output');}catch(e){if(e.code!=='ENOENT')throw e;}
const w=Workbook.create();const s=w.worksheets.add('统一结果');s.showGridLines=false;
s.getRange('A1:L18').format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:30,verticalAlignment:'center'};
s.getRange('A1:L14').values=[data.headers,...data.rows];
s.getRange('A1:L1').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:44};
const widths=[27,35,10,8,18,18,18,18,27,22,22,27];
for(let i=0;i<widths.length;i++)s.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=widths[i];
s.getRange('C2:L14').format.horizontalAlignment='right';
s.getRange('C2:D14').setNumberFormat('0');s.getRange('E2:I14').setNumberFormat('#,##0.000');s.getRange('J2:L14').setNumberFormat('0.000"x"');
s.getRange('A2:L2').format.fill='#EAF1F7';s.getRange('B2').format.font.bold=true;s.getRange('E2').format.font.bold=true;
for(const [i,note] of data.notes.entries()){
  s.getRange('A'+(16+i)).values=[[note]];
  s.getRange(`A${16+i}:L${16+i}`).format={font:{size:10,color:'#526879'},rowHeight:25};
}
const p=data.phase_table;
if(!p||p.rows.length!==20||p.rounds.length!==3)throw Error('Complete source-derived phase table required');
const t=w.worksheets.add('LightGen时间分块');t.showGridLines=false;
t.getRange('A1:F36').format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:42,verticalAlignment:'center'};
for(const [i,width] of [53,23,23,23,23,98].entries())t.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=width;
t.getRange('A1:F21').values=[p.headers,...p.rows];
t.getRange('A1:F1').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:55};
t.getRange('A2:F21').format.wrapText=true;
t.getRange('B2:E21').format.horizontalAlignment='right';t.getRange('B2:E21').setNumberFormat('0.000000');
t.getRange('A21:F21').format={fill:'#EAF1F7',font:{bold:true}};
for(const [i,note] of p.notes.entries()){
  t.getRange('A'+(24+i)).values=[[note]];
  t.getRange(`A${24+i}:F${24+i}`).format={font:{size:10,color:'#526879'},rowHeight:25,wrapText:false};
}
t.getRange('A33:E36').values=[p.round_headers,...p.rounds];
t.getRange('A33:E33').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:32};
t.getRange('B34:E36').format.horizontalAlignment='right';t.getRange('A34:B36').setNumberFormat('0');t.getRange('C34:E36').setNumberFormat('0.000000');
w.recalculate();
if(s.getRange('A1:L18').formulas.flat().some(Boolean))throw Error('No formulas permitted');
if(t.getRange('A1:F36').formulas.flat().some(Boolean))throw Error('No phase formulas permitted');
console.log((await w.inspect({kind:'table',range:'统一结果!A1:L4',include:'values,formulas',tableMaxRows:4,tableMaxCols:12,maxChars:1500})).ndjson);
await fs.mkdir(out);
const target=path.join(out,'Temporal_LightGen与baseline_固定数值_20261027.xlsx');
await (await SpreadsheetFile.exportXlsx(w)).save(target);
const preview=await w.render({sheetName:'统一结果',range:'A1:L18',scale:1.3,format:'png'});
await fs.writeFile(path.join(out,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
for(const [name,range] of [['phase_preview','A1:F21'],['phase_notes','A23:F36']]){
  const blob=await w.render({sheetName:'LightGen时间分块',range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
await fs.copyFile(input,path.join(out,'static_values.json'));
console.log(JSON.stringify({target,rows:13,sheets:2,formulas:0}));
