/** Targeted LSP main-sheet edit; preserve the existing sample sheet and user edits. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
const [file,authorDir,mode]=process.argv.slice(2);
const require=createRequire(path.join(path.resolve(authorDir),'package.json'));
const {FileBlob,Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const verification=path.join(path.dirname(file),'verification');await fs.mkdir(verification,{recursive:true});
const backup=path.join(verification,'lsp_before_joint_merge.bin');
const onDisk=await fs.readFile(file),digest=v=>createHash('sha256').update(v).digest('hex');
const fromBackup=['format-repair','chart-layout'].includes(mode);
if(fromBackup){
  const prior=JSON.parse(await fs.readFile(path.join(verification,'lsp_joint_merge_audit.json'),'utf8'));
  if(prior.after_SHA!==digest(onDisk))throw Error('Workbook changed since our edit; refuse overwrite.');
}
const raw=fromBackup?await fs.readFile(backup):onDisk;
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(fromBackup?backup:file));
const sh=wb.worksheets.getItem('整体指标'),samples=wb.worksheets.getItem('逐测试样本');
if(mode==='inspect'){
  const image=await wb.render({sheetName:sh.name,range:'A2:E40',scale:1,format:'png'});
  await fs.writeFile(path.join(verification,'lsp_before_joint_merge.png'),new Uint8Array(await image.arrayBuffer()));
  console.log((await wb.inspect({kind:'table',sheetId:sh.name,range:'A13:E16',include:'values',tableMaxRows:4,tableMaxCols:5,maxChars:1200})).ndjson);
  process.exit(0);
}
const before=sh.getRange('A1:H40').values,sampleBefore=samples.getUsedRange().values;
const scores=new Map(before.slice(12,26).map(r=>[r[1],r[2]]));
const counts=new Map(before.slice(26,40).map(r=>[r[1],r[2]]));
if(scores.size!==14||counts.size!==14)throw Error('Expected original14-joint summary; refuse a second or ambiguous edit.');
const groups=[['ankle','踝关节'],['knee','膝关节'],['hip','髋关节'],['wrist','腕关节'],['elbow','肘关节'],['shoulder','肩关节'],['neck','颈部'],['head_top','头顶']];
const merged=groups.map(([name,label])=>{
  const keys=scores.has(name)?[name]:['left_'+name,'right_'+name];
  if(keys.some(k=>!scores.has(k)||!counts.has(k)))throw Error('Missing source joint '+name);
  const count=keys.reduce((n,k)=>n+counts.get(k),0),hits=keys.reduce((n,k)=>n+scores.get(k)*counts.get(k),0);
  return {name,label,sourceJoints:keys,count,hits,pck:hits/count};
});
const total=merged.reduce((n,r)=>n+r.count,0),global=merged.reduce((n,r)=>n+r.hits,0)/total;
if(total!==before[9][2]||Math.abs(global-before[5][2])>1e-12)throw Error('Weighted joint summary disagrees with original overall PCK.');
const rows=[...merged.map(r=>['关节类型PCK',r.label,r.pck,'无量纲','越大越好']),...merged.map(r=>['关节类型样本数',r.label,r.count,'个','—'])];
const table=sh.tables.items.find(t=>t.name==='OverallMetrics');if(!table)throw Error('Original metric table missing');
table.delete();
let changedStart=12,expectedRange='A13:E28',expectedRows=rows,tableRange='A5:E28';
if(mode==='chart-layout'){
  changedStart=4;tableRange='A5:C12';expectedRange='A6:C12';
  const order=['shoulder','elbow','wrist','hip','knee','ankle'];
  expectedRows=[['Overall (14 joints)',global,total],...order.map(name=>{const g=merged.find(r=>r.name===name);return [name[0].toUpperCase()+name.slice(1),g.pck,g.count];})];
  sh.getRange('A5:E40').clear({applyTo:'all'});
  sh.getRange('A5:C12').format={font:{name:'Arial',size:11,color:'#1C3041'},verticalAlignment:'center',rowHeight:25};
  sh.getRange('A5:C5').values=[['部位','PCK@0.2','有效关节数']];
  sh.getRange('A6:C12').values=expectedRows;
  sh.getRange('B6:B12').setNumberFormat('0.000000');sh.getRange('C6:C12').setNumberFormat('#,##0');
  sh.getRange('B6:C12').format.horizontalAlignment='right';
  sh.getRange('A5:C5').format={fill:'#066CB1',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:28};
  for(let row=6;row<=12;row++)sh.getRange(`A${row}:C${row}`).format.fill=row%2===0?'#EEF4F8':'#FFFFFF';
  sh.getRange('A6:C6').format.font={name:'Arial',size:11,bold:true,color:'#124C75'};
  sh.getRange('A16:C16').values=[['其他整体指标','数值','单位']];
  sh.getRange('A17:C22').values=before.slice(6,12).map(r=>[r[1],r[2],r[3]]);
  sh.getRange('A16:C22').format={font:{name:'Arial',size:11,color:'#1C3041'},verticalAlignment:'center',rowHeight:25};
  sh.getRange('A16:C16').format={fill:'#066CB1',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:28};
  sh.getRange('B17:B19').setNumberFormat('0.000000');sh.getRange('B20:B22').setNumberFormat('#,##0');
  for(let row=17;row<=22;row++)sh.getRange(`A${row}:C${row}`).format.fill=row%2===1?'#EEF4F8':'#FFFFFF';
  sh.getRange('A25').values=[['Overall含14关节；六类部位左右合并，颈部和头顶不单列。']];
  sh.getRange('A25:E25').format.font={name:'Arial',size:10,color:'#5B6C78'};
}else{
sh.getRange('A13:E40').clear({applyTo:'contents'});
sh.getRange('A29:E40').clear({applyTo:'formats'});
sh.getRange('A13:E28').values=rows;
sh.getRange('C13:C20').setNumberFormat('0.000000');sh.getRange('C21:C28').setNumberFormat('#,##0');
sh.getRange('C13:C28').format.horizontalAlignment='right';
for(let row=13;row<=28;row++)sh.getRange(`A${row}:E${row}`).format.fill=row%2===0?'#EEF4F8':'#FFFFFF';
}
const revised=sh.tables.add(tableRange,true,'OverallMetrics');revised.style='TableStyleMedium2';
wb.recalculate();
if(!isDeepStrictEqual(samples.getUsedRange().values,sampleBefore))throw Error('Untouched sample sheet changed.');
const after=sh.getRange('A1:H40').values;
for(let r=0;r<40;r++)for(let c=0;c<8;c++)if(!(r>=changedStart&&c<5)&&!isDeepStrictEqual(before[r][c],after[r][c]))throw Error('Unrequested main-sheet cell changed.');
if(!isDeepStrictEqual(sh.getRange(expectedRange).values,expectedRows))throw Error('Merged rows changed.');
const image=await wb.render({sheetName:sh.name,range:'A2:E28',scale:1,format:'png'});
await fs.writeFile(path.join(verification,'lsp_after_joint_merge.png'),new Uint8Array(await image.arrayBuffer()));
try{await fs.writeFile(backup,raw,{flag:'wx'});}catch(e){if(e.code!=='EEXIST')throw e;if(digest(await fs.readFile(backup))!==digest(raw))throw Error('Distinct earlier edit backup exists.');}
if(digest(await fs.readFile(file))!==digest(onDisk))throw Error('Workbook changed during edit; refuse overwrite.');
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(file);
const audit={status:'passed',before_SHA:digest(raw),after_SHA:digest(await fs.readFile(file)),groups:merged,pooledPCK:global,pooledJoints:total,
  mainOverallValuesUnchanged:true,otherMainCellsUnchanged:true,sampleSheetValuesUnchanged:true,changedRange:`整体指标!A${changedStart+1}:E40`,tableRange,chartOrder:mode==='chart-layout'?expectedRows.map(r=>r[0]):null,formula:'sum(PCK_j * count_j) / sum(count_j)',newModelForwards:0};
await fs.writeFile(path.join(verification,'lsp_joint_merge_audit.json'),JSON.stringify(audit,null,2));
console.log(JSON.stringify(audit));
