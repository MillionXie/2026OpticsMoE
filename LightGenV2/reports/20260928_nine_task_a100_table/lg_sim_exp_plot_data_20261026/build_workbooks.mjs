/** Selected Sim/Exp plotting data only. Original runs and evidence are untouched. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
const out=path.dirname(fileURLToPath(import.meta.url));
const require=createRequire(path.resolve('.codex_tmp/package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload=JSON.parse(await fs.readFile(path.join(out,'plot_data.json'),'utf8'));
const verify=path.join(out,'verification');await fs.mkdir(verify,{recursive:true});
const chosen=new Set(process.argv.slice(2));
const previous=chosen.size?JSON.parse(await fs.readFile(path.join(verify,'workbook_export_audit.json'),'utf8')).files:[];
const colName=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const isCount=k=>/^(samples|queries|candidates|count|evaluated_joints|pck_evaluated_joints|pckh_evaluated_joints)$/.test(k);
const unit=(k,d)=>k==='psnr_db'?'dB':isCount(k)?'个':k.includes('pixel_error')?'px':d.task.startsWith('lgvq')&&['mae','rmse'].includes(k)?'MOS':['srcc','krcc','plcc','cc'].includes(k)?'相关系数':'无量纲';
const direction=k=>isCount(k)?'—':/mae|rmse|mse|kld|error/.test(k)?'越小越好':'越大越好';
const groupName=g=>g==='overall'?'Overall':g.startsWith('PCK group/')?g.slice(10):g.startsWith('mode/')?g.slice(5):g==='overall/per_joint_counts'?'Joint count':g==='overall/per_joint_pck_at_0.2_torso'?'Joint PCK@0.2':g;
const font={name:'Arial',size:11,color:'#203746'};
const header={fill:'#116A9D',font:{...font,bold:true,color:'#FFFFFF'},rowHeight:32,verticalAlignment:'center',wrapText:true};
const audit=[];
for(const d of payload.tasks){
  if(chosen.size&&!chosen.has(d.task))continue;
  const view=d.plot_view;if(!view?.selected_results_only)throw Error('Missing selected-only projection '+d.task);
  const wb=Workbook.create(),main=wb.worksheets.add('整体指标'),raw=wb.worksheets.add('逐测试样本');
  main.showGridLines=false;raw.showGridLines=false;main.tabColor='#116A9D';raw.tabColor='#D59C50';
  const maps=Object.fromEntries(['Sim','Exp'].map(s=>[s,new Map(view.metrics[s].map(([g,k,v])=>[g+'\t'+k,v]))]));
  const keys=[...new Set([...d.required_metrics.map(k=>'overall\t'+k),...maps.Sim.keys(),...maps.Exp.keys()])];
  if(d.task==='lsp')keys.sort((a,b)=>(a.startsWith('PCK group/')?0:1)-(b.startsWith('PCK group/')?0:1));
  const values=keys.map(key=>{const [g,k]=key.split('\t');return [groupName(g),k,maps.Sim.get(key)??null,maps.Exp.get(key)??null,g.endsWith('counts')?'个':unit(k,d),g.endsWith('counts')?'—':direction(k)];});
  const end=3+values.length;
  main.getRange(`A1:F${end}`).format={font,rowHeight:25,verticalAlignment:'center'};
  for(const [i,w] of [29,42,20,20,16,18].entries())main.getRangeByIndexes(0,i,end,1).format.columnWidth=w;
  main.getRange('A1:F1').merge();main.getRange('A1').values=[[d.name+' — LightGen-II']];
  main.getRange('A1:F1').format={font:{...font,size:15,bold:true,color:'#174D70'},rowHeight:36};
  main.getRange('A3:F3').values=[['分组','指标','Sim.','Exp.','单位','方向']];main.getRange('A3:F3').format=header;
  main.getRange(`A4:F${end}`).values=values;main.getRange(`C4:D${end}`).setNumberFormat('0.000000');
  main.getRange(`C4:D${end}`).format.horizontalAlignment='right';
  main.tables.add(`A3:F${end}`,true,'OverallMetrics');
  for(let i=0;i<values.length;i++){
    const row=i+4,[g,k]=keys[i].split('\t');
    if(isCount(k)||g.endsWith('counts'))main.getRange(`C${row}:D${row}`).setNumberFormat('#,##0');
    if(g==='overall'&&k===d.primary)main.getRange(`A${row}:F${row}`).format={fill:'#DCECF5',font:{...font,bold:true,color:'#164D6D'}};
  }
  if(end>25)main.freezePanes.freezeRows(3);
  const n=view.samples.length,c=view.headers.length;
  raw.getRangeByIndexes(0,0,Math.max(n+1,2),c).format={font,rowHeight:25,verticalAlignment:'center'};
  raw.getRangeByIndexes(0,0,1,c).values=[view.headers];raw.getRangeByIndexes(0,0,1,c).format={...header,rowHeight:48};
  if(n){raw.getRangeByIndexes(1,0,n,c).values=view.samples;raw.tables.add(`A1:${colName(c-1)}${n+1}`,true,'SavedTestSamples');}
  let detailHeight=25;
  for(let i=0;i<c;i++){
    const key=view.headers[i],first=view.samples.find(r=>r[i]!==null)?.[i];
    const wrap=/instruction|prompt|^title$|positive_ranks/.test(key);
    const width=i===0?12:key==='sample_id'&&d.task.startsWith('lgvq')?80:wrap?64:/sample_id|query_id|source_id/.test(key)?32:25;
    raw.getRangeByIndexes(0,i,Math.max(n+1,2),1).format.columnWidth=width;
    if(n&&wrap){raw.getRangeByIndexes(1,i,n,1).format.wrapText=true;detailHeight=Math.max(detailHeight,key==='positive_ranks'&&d.task==='abo_image_to_image'?28:48);}
    if(n&&typeof first!=='string'){
      raw.getRangeByIndexes(1,i,n,1).setNumberFormat(/rank|count|joints|_hit|^H@/.test(key)?'0':'0.000000');
      raw.getRangeByIndexes(1,i,n,1).format.horizontalAlignment='right';
    }
  }
  if(n)raw.getRangeByIndexes(1,0,n,c).format.rowHeight=detailHeight;
  raw.freezePanes.freezeRows(1);raw.freezePanes.freezeColumns(2);
  wb.recalculate();
  if(!isDeepStrictEqual(main.getRange(`A4:F${end}`).values,values))throw Error('Summary readback mismatch '+d.task);
  if(n&&!isDeepStrictEqual(raw.getRangeByIndexes(1,0,n,c).values,view.samples))throw Error('Sample readback mismatch '+d.task);
  const inspected=await wb.inspect({kind:'region',sheetId:main.name,range:`A3:F${Math.min(end,12)}`,maxChars:2500});
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000});
  if(/"matchCount"\s*:\s*[1-9]/.test(errors.ndjson))throw Error(errors.ndjson);
  for(const [sheet,range,suffix] of [[main,`A1:F${end}`,'overall'],[main,`A1:F${Math.min(end,32)}`,'overall_top'],[raw,`A1:${colName(Math.min(c,8)-1)}9`,'samples'],[raw,`E1:${colName(c-1)}9`,'metric_details']]){
    const pic=await wb.render({sheetName:sheet.name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(verify,d.task+'_'+suffix+'.png'),new Uint8Array(await pic.arrayBuffer()));
  }
  const expFirst=Math.max(2,view.samples.findIndex(r=>r[0]==='Exp')+2);
  const expPic=await wb.render({sheetName:raw.name,range:`A${expFirst}:${colName(Math.min(c,12)-1)}${expFirst+7}`,scale:1,format:'png'});
  await fs.writeFile(path.join(verify,d.task+'_samples_exp.png'),new Uint8Array(await expPic.arrayBuffer()));
  const file=path.join(out,d.filename);await(await SpreadsheetFile.exportXlsx(wb)).save(file);
  const bytes=await fs.readFile(file);
  audit.push({task:d.task,filename:d.filename,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,sheets:2,summary_start_row:4,
    source_sample_rows:n,archived_reference_rows:view.excluded_reference_rows,view:'selected_plot_only',full_in_memory_readback:true,
    inspection:inspected.ndjson,error_scan:errors.ndjson,summary_values:values});
  console.log(JSON.stringify({task:d.task,file:d.filename,sample_rows:n,excluded_reference_rows:view.excluded_reference_rows,bytes:bytes.length}));
}
const merged=payload.tasks.map(d=>audit.find(x=>x.task===d.task)??previous.find(x=>x.task===d.task));
if(merged.some(x=>!x))throw Error('Missing prior workbook audit in scoped update');
await fs.writeFile(path.join(verify,'workbook_export_audit.json'),JSON.stringify({date:payload.date,workbooks:merged.length,files:merged},null,2));
console.log('Completed '+audit.length+' selected-only workbooks.');
