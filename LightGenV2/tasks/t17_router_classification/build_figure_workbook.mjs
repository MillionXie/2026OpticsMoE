import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out=path.resolve(process.argv[2]);
const preview=path.join(path.dirname(out),'figure_previews_20261101');
const d=JSON.parse(await fs.readFile(path.join(out,'data/source_data.json'),'utf8'));
await fs.mkdir(preview,{recursive:true});
const wb=Workbook.create();const expected=[];const views=[];
const letter=i=>String.fromCharCode(65+i);
function make(name,title,note,headers,rows,widths){
  const s=wb.worksheets.add(name);s.showGridLines=false;
  const last=letter(headers.length-1);const end=5+rows.length;
  s.getRange(`A1:${last}${end+2}`).format.font={name:'Arial',size:11,color:'#18324B'};
  s.getRange(`A1:${last}${end+2}`).format.rowHeight=25;
  headers.forEach((_,i)=>{s.getRange(`${letter(i)}1:${letter(i)}${end+2}`).format.columnWidthPx=widths[i]||130;});
  s.getRange('A2').values=[[title]];s.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#173E68'};s.getRange('A2').format.rowHeight=31;
  s.getRange('A3').values=[[note]];s.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#596C7D'};
  s.getRange(`A5:${last}${end}`).values=[headers,...rows];
  s.getRange(`A5:${last}5`).format={fill:'#173E68',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:32,horizontalAlignment:'center',verticalAlignment:'center'};
  for(let r=6;r<=end;r++){if(r%2===0)s.getRange(`A${r}:${last}${r}`).format.fill='#F0F4F8';}
  s.getRange(`A6:${last}${end}`).format.verticalAlignment='center';
  if(rows.length>10)s.freezePanes.freezeRows(5);
  views.push({sheet:name,range:`A1:${last}${end+2}`});return s;
}
function formula(s,address,text,value){s.getRange(address).formulas=[[text]];expected.push({sheet:s.name,address,value});}

const notes=make('Notes','A B D E F drawing handoff','2026-11-01 / fixed original TEST development / seed 17', ['项目','交接口径'],d.notes,[180,1080]);
notes.getRange(`B6:B${5+d.notes.length}`).format.wrapText=true;notes.getRange(`A6:B${5+d.notes.length}`).format.rowHeight=48;

const a=make('A_Classification','A  Classification','Source record IDs resolve to data/source_data.json; rates are correct / original N.',
 ['Dataset','Model','Layers','TEST N','Correct','TEST dev','TRAIN','VAL','OEO','Linear','Record ID'],
 d.A.map(v=>[v.dataset,v.model,v.layers,v.test.n,v.test.correct,null,v.train?.accuracy??'n.a.',v.val?.accuracy??'n.a.',v.OEO,v.Linear,v.record]),[170,105,80,95,95,115,115,115,80,80,240]);
d.A.forEach((v,i)=>formula(a,`F${6+i}`,`=E${6+i}/D${6+i}`,v.test.accuracy));a.getRange('F6:H11').setNumberFormat('0.00%');

const b=make('B_Depth','B  Kather2016 depth','LG2 shares the A Kather / D k2 checkpoint. LG6 gap is 14.76 pp, below the original 15 pp goal.',
 ['LG layers','D2NN layers','TEST N','LG correct','D correct','LG TEST','D TEST','Gap (pp)','LG record','D record'],
 d.B.map(v=>[v.LG_layers,v.D2NN_layers,v.n,v.LG_correct,v.D2NN_correct,null,null,null,v.LG_record,v.D2NN_record]),[100,115,95,110,110,120,120,120,240,200]);
d.B.forEach((v,i)=>{const r=i+6;formula(b,`F${r}`,`=D${r}/C${r}`,v.LG_accuracy);formula(b,`G${r}`,`=E${r}/C${r}`,v.D2NN_accuracy);formula(b,`H${r}`,`=(F${r}-G${r})*100`,v.gap_pp);});
b.getRange('F6:G8').setNumberFormat('0.00%');b.getRange('H6:H8').setNumberFormat('0.00');b.getRange('H6:H8').conditionalFormats.add('cellIs',{operator:'lessThan',formula:15,format:{fill:'#FFF1CE',font:{color:'#8E4A00'}}});

const ds=make('D_TopK','D  Actual Top-k','Kather main curve is a saved-PT constrained tuple; independent maxima: 601 / 596 / 604 / 604 (N=752).',
 ['Dataset','Top-k','TEST N','Correct','TEST dev','PT selection','Record ID'],
 d.D.map(v=>[v.dataset,v.k,v.n,v.correct,null,v.selection,v.record]),[170,80,110,110,120,245,260]);
d.D.forEach((v,i)=>formula(ds,`E${6+i}`,`=D${6+i}/C${6+i}`,v.accuracy));ds.getRange('E6:E17').setNumberFormat('0.00%');

const e=make('E_ExpertUsage','E  Same-PT expert usage','Main E: first 8 TEST_DEV rows. Share=count/(2N); frequency=count/N. Power is a separate measure.',
 ['Split','Modality','Expert','Images N','Selections','Frequency','Slot share','Post power','Dense power','Record ID'],
 d.E.map(v=>[v.split,v.modality,'E'+v.expert,v.n,v.selected,null,null,v.post_power,v.dense_power,v.record]),[110,100,85,110,115,120,120,120,125,240]);
d.E.forEach((v,i)=>{const r=6+i;formula(e,`F${r}`,`=E${r}/D${r}`,v.frequency);formula(e,`G${r}`,`=E${r}/(2*D${r})`,v.share);});e.getRange('F6:I29').setNumberFormat('0.00%');

const f=make('F_Modalities','F  Joint LG and source-only D2NN','Six main values: one joint LG x two modalities; two source-only D2NNs x both modalities. Same LG PT for every LG metric.',
 ['Model','Trained on','Infer on','TEST N','Correct','TEST dev','Record ID'],
 d.F.map(v=>[v.model,v.train_domain,v.test_domain,v.n,v.correct,null,v.record]),[110,175,115,110,110,130,270]);
d.F.forEach((v,i)=>formula(f,`F${6+i}`,`=E${6+i}/D${6+i}`,v.accuracy));f.getRange('F6:F11').setNumberFormat('0.00%');
f.getRange('A14').values=[['Same LG PT: original TRAIN / VAL / TEST diagnostics']];f.getRange('A14').format.font={bold:true,size:12,color:'#173E68'};
f.getRange('A16:J18').values=[['Modality','TRAIN N','TRAIN correct','TRAIN acc','VAL N','VAL correct','VAL acc','TEST N','TEST correct','TEST dev'],...['RGB','SAR'].map(domain=>{const v=d.F_joint_training[domain];return[domain,v.train.n,v.train.correct,null,v.val.n,v.val.correct,null,v.test.n,v.test.correct,null];})];
f.getRange('A16:J16').format={fill:'#173E68',font:{name:'Arial',bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:32};f.getRange('A14:J20').format.font.name='Arial';
for(let i=0;i<2;i++){const r=i+17;const v=d.F_joint_training[['RGB','SAR'][i]];formula(f,`D${r}`,`=C${r}/B${r}`,v.train.accuracy);formula(f,`G${r}`,`=F${r}/E${r}`,v.val.accuracy);formula(f,`J${r}`,`=I${r}/H${r}`,v.test.accuracy);}
for(const col of ['D','G','J'])f.getRange(`${col}17:${col}18`).setNumberFormat('0.00%');
f.getRange('H1:J20').format.columnWidthPx=125;views[views.length-1].range='A1:J20';

wb.recalculate();
for(const v of expected){const actual=wb.worksheets.getItem(v.sheet).getRange(v.address).values[0][0];if(typeof actual!=='number'||Math.abs(actual-v.value)>1e-10)throw new Error(`Mismatch ${v.sheet}!${v.address}: ${actual} versus ${v.value}`);}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:1500});
console.log(errors.ndjson);
console.log((await wb.inspect({kind:'table',range:'F_Modalities!A5:F11',include:'values,formulas',tableMaxRows:7,tableMaxCols:6,maxChars:1800})).ndjson);
for(const v of views){const png=await wb.render({sheetName:v.sheet,range:v.range,scale:1,format:'png'});await fs.writeFile(path.join(preview,v.sheet+'.png'),new Uint8Array(await png.arrayBuffer()));}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path.join(out,'Figure_data.xlsx'));
await fs.writeFile(path.join(preview,'verification.json'),JSON.stringify({formula_checks:expected.length,all_numeric_formulas_match_source:true,sheets:views,error_scan:errors.ndjson},null,2));
console.log(JSON.stringify({workbook:path.join(out,'Figure_data.xlsx'),preview,formula_checks:expected.length,sheets:views.length}));
