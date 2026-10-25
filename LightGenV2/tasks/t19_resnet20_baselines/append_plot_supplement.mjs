/** Append requested data to the two user-edited main sheets, preserving originals. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
const [directory,payloadPath,authorDir,mode]=process.argv.slice(2);
const req=createRequire(path.join(path.resolve(authorDir),'package.json'));
const {FileBlob,SpreadsheetFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const digest=raw=>createHash('sha256').update(raw).digest('hex');
const payloadRaw=await fs.readFile(payloadPath),data=JSON.parse(payloadRaw);
if(data.status!=='passed'||data.new_neural_forwards!==0)throw Error('Unverified supplement');
const verification=path.join(directory,'verification');
await fs.mkdir(verification,{recursive:true});
const reports=[];
for(const [key,name] of [['abo','04_ABO图搜图.xlsx'],['salicon','07_SALICON显著性.xlsx']]){
 const file=path.join(directory,name),onDisk=await fs.readFile(file);
 let raw=onDisk,importPath=file;
 const backup=path.join(verification,key+'_before_plot_supplement.bin');
 const auditFile=path.join(verification,key+'_plot_supplement_audit.json');
 try{
  const prior=JSON.parse(await fs.readFile(auditFile,'utf8'));
  if(prior.after_SHA!==digest(onDisk)||prior.payload_SHA!==digest(payloadRaw))throw Error('Workbook changed after previous supplement; refuse overwrite');
  if(mode==='repair-layout'&&key==='salicon'){
   raw=await fs.readFile(backup);importPath=backup;
   if(prior.before_SHA!==digest(raw))throw Error('Original backup identity differs');
  }else{reports.push(prior);continue;}
 }catch(e){if(e.code!=='ENOENT')throw e;}
 const beforeSHA=digest(raw);
 const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(importPath));
 const main=wb.worksheets.getItem('整体指标'),sample=wb.worksheets.getItem('逐测试样本');
 const beforeMain=main.getRange('A1:P40').values,beforeSamples=sample.getUsedRange().values;
 const blank=range=>{if(main.getRange(range).values.some(row=>row.some(v=>v!==null&&v!=='')))throw Error('Requested target has existing user content: '+range);};
 const noteFont={name:'Arial',size:10,color:'#5B6C78'};
 const headerFont={name:'Arial',size:11,bold:true,color:'#FFFFFF'};
 let expected,range,renderRange,allowed;
 if(key==='abo'){
  blank('A15:E31');
  main.getRange('A15').values=[['按查询所属类别统计 · 精确SKU Top-1']];
  main.getRange('A15:E15').format.font={name:'Arial',size:12,bold:true,color:'#124C75'};
  main.getRange('A16:D16').values=[['Category','Queries','Correct Top-1','H@1 (%)']];
  expected=data.abo_categories.rows;range='A17:D26';
  if(expected.length!==10||expected.reduce((n,r)=>n+r[2],0)!==623)throw Error('Category totals differ');
  main.getRange(range).values=expected;
  const table=main.tables.add('A16:D26',true,'ResNetCategoryAccuracy');table.style='TableStyleMedium2';
  main.getRange('A16:D26').format.font={name:'Arial',size:11,color:'#1C3041'};
  main.getRange('A16:D16').format={fill:'#066CB1',font:headerFont,horizontalAlignment:'center'};
  main.getRange('B17:C26').setNumberFormat('#,##0');
  main.getRange('D17:D26').setNumberFormat('0.00%');
  main.getRange('B17:D28').format.horizontalAlignment='right';
  main.getRange('A28:D28').values=[['Overall',...data.abo_categories.overall]];
  main.getRange('A28:D28').format={fill:'#DCECF6',font:{name:'Arial',size:11,bold:true,color:'#124C75'}};
  main.getRange('B28:C28').setNumberFormat('#,##0');main.getRange('D28').setNumberFormat('0.00%');
  main.getRange('A30').values=[['按图示类别顺序；每类80查询、20商品。命中要求同一SKU，非只命中同类。']];
  main.getRange('A31').values=[['H@1保留0–1数值，单元格显示为百分比；零任务训练的最终固定权重。']];
  main.getRange('A30:E31').format.font=noteFont;
  renderRange='A15:E31';allowed=(r,c)=>r>=14&&r<=30&&c<5;
 }else{
  blank('G2:K5005');blank('A16:E22');
  main.getRange('G2').values=[['完整逐图分布数据 · 5,000 TEST样本']];
  main.getRange('G2:K2').format.font={name:'Arial',size:12,bold:true,color:'#124C75'};
  main.getRange('G3').values=[['每行对应同一张图，可直接用于MAE / KLD / CC / NSS分布图。']];
  main.getRange('G3:K3').format.font=noteFont;
  main.getRange('G5:K5').values=[['sample_id','MAE ↓','KLD ↓','CC ↑','NSS ↑']];
  expected=data.saliency.rows.map(r=>r.slice(0,5));range='G6:K5005';
  if(expected.length!==5000||new Set(expected.map(r=>r[0])).size!==5000)throw Error('Incomplete SALICON samples');
  main.getRange(range).values=expected;
  const table=main.tables.add('G5:K5005',true,'ResNetPerImageDistributions');table.style='TableStyleMedium2';
  main.getRange('G5:K5005').format.font={name:'Arial',size:11,color:'#1C3041'};
  main.getRange('G5:K5').format={fill:'#066CB1',font:headerFont,horizontalAlignment:'center'};
  // Artifact columnWidth uses Excel character units, not pixels.
  main.getRange('G:K').format.columnWidth=16;main.getRange('G:G').format.columnWidth=32;
  // Rows13 onward were blank after the user's metadata removal; normalize
  // their inherited tall heights for the newly requested long sample table.
  main.getRange('G13:K5005').format.rowHeight=25;
  main.getRange('H6:K5005').setNumberFormat('0.000000');main.getRange('H6:K5005').format.horizontalAlignment='right';
  main.freezePanes.freezeRows(5);
  const notes=[
   '主表右侧G:K：完整5000个TEST样本，保留负值和分布尾部。',
   '样本ID逐行对齐；MAE、KLD越低越好，CC、NSS越高越好。',
   'MAE：预测与标签分别按峰值归一化后的逐像素绝对误差均值。',
   'KLD：KL(标签 || 预测)，自然对数；CC保留原逐图保存值。',
   'NSS：在真实注视点上取预测标准分数的均值（总体标准差）。',
   '由封存的7×7预测图按原224×224评估口径复算，无新增模型推理。',
   '逐图均值与左侧FP32批次聚合量只有舍入差；不按图中坐标截断。',
  ];
  main.getRange('A16:A22').values=notes.map(n=>[n]);main.getRange('A16:E22').format.font=noteFont;
  renderRange='G2:K17';allowed=(r,c)=>(c>=6&&c<=10&&r>=1&&r<=5004)||(c<5&&r>=15&&r<=21);
 }
 wb.recalculate();
 const afterMain=main.getRange('A1:P40').values;
 for(let r=0;r<40;r++)for(let c=0;c<16;c++)if(!allowed(r,c)&&!isDeepStrictEqual(beforeMain[r][c],afterMain[r][c]))throw Error('Unrequested main-sheet value changed');
 if(!isDeepStrictEqual(beforeSamples,sample.getUsedRange().values))throw Error('Original sample sheet changed');
 if(!isDeepStrictEqual(main.getRange(range).values,expected))throw Error('Inserted data do not match payload');
 const image=await wb.render({sheetName:main.name,range:renderRange,scale:1.2,format:'png'});
 await fs.writeFile(path.join(verification,key+'_after_plot_supplement.png'),new Uint8Array(await image.arrayBuffer()));
 try{await fs.writeFile(backup,raw,{flag:'wx'});}catch(e){if(e.code!=='EEXIST'||digest(await fs.readFile(backup))!==beforeSHA)throw e;}
 const temp=path.join(verification,key+'_supplement_candidate.xlsx');
 const output=await SpreadsheetFile.exportXlsx(wb);await output.save(temp);
 if(digest(await fs.readFile(file))!==digest(onDisk))throw Error('Workbook changed during edit; refuse overwrite');
 await fs.copyFile(temp,file);await fs.unlink(temp);
 const audit={status:'passed',file:name,before_SHA:beforeSHA,after_SHA:digest(await fs.readFile(file)),payload_SHA:digest(payloadRaw),
  insertedRange:range,insertedRows:expected.length,originalMainValuesPreserved:true,originalSampleSheetValuesPreserved:true,
  newModelForwards:0,backup:path.basename(backup)};
 await fs.writeFile(auditFile,JSON.stringify(audit,null,2));reports.push(audit);
}
console.log(JSON.stringify(reports));
