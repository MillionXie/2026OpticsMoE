/** Author nine archival workbooks with artifact-tool. Does not edit source runs. */
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
const statusLabels={verified_full_test:'原报告与样本已核对',verified_mixed_test:'混合旧TEST；非独立测试',same_pt_packaged_replay:'同PT重放；与历史末位有差异',aggregate_only:'仅聚合；逐样本待找回',not_located:'截图版本原报告未定位',index_only_pending_raw:'仅版本索引；原始排名待取回',verified_full_queries:'完整逐查询已核对',verified_saved_predictions:'已保存预测与原指标已核对',numeric_match_lineage_unconfirmed:'数值吻合；截图身份未证实',screenshot_rounding_mismatch:'原记录与截图四舍五入不符',aggregate_only_unpaired_samples:'仅聚合；无实验逐样本ID',verified_saved_metrics:'原逐样本指标已核对'};
Object.assign(statusLabels,{source_found_inconsistent:'原报告已找回；汇总与明细不符',verified_saved_top1:'原800条Top-1与原汇总已核对',verified_user_selected_adapted:'用户已明确确认采用的实验适配版',aggregate_with_verified_sample_identity:'样本身份已核对；逐例结果未导出'});
Object.assign(statusLabels,{verified_corrected_saved_summary:'用户确认校正；逐条与embedding复算一致',verified_frozen_CCD_replay:'原CCD固定权重补算；整体/分组已核对'});
statusLabels.same_pt_packaged_replay='同PT回放；与历史训练另列';
Object.assign(statusLabels,{original_final_report:'指定原最终报告已核；原排名待恢复',final_identity_unconfirmed:'最终实验身份未确认；旧会话不代用'});
const count=k=>/^(samples|queries|candidates|count|gallery_count|query_count|candidate_count|relevant_images_per_query|relevant_candidates_per_query|evaluated_joints|pck_evaluated_joints|pckh_evaluated_joints)$/.test(k);
const unit=(k,d)=>k==='psnr_db'?'dB':count(k)?'个':k.includes('pixel_error')?'px':d.task.startsWith('lgvq')&&['mae','rmse'].includes(k)?'MOS分':['srcc','krcc','plcc','cc'].includes(k)?'相关系数':'无量纲';
const dir=k=>count(k)||k==='epoch'?'—':/loss|mae|rmse|mse|kld|error|mean_rank|median_rank/.test(k)?'越小越好':'越大越好';
const label=g=>g==='overall'?'整体':g==='same_pt_replay_reference'?'同PT回放参考（非指定版）':g==='superseded_session_reference'?'旧会话参考（非最终）':g==='historical_training'?'历史训练记录':g==='historical_FP32_summary'?'原FP32汇总':g==='original_reported_summary'?'原报告汇总（校正前）':g==='CPU_replay_validation'?'固定权重CPU回放':g.startsWith('PCK group/')?g.slice(10):g.startsWith('category/')?'类别/'+g.slice(9):g.startsWith('mode/')?'编辑类型/'+g.slice(5):g;
const colName=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const header={fill:'#116A9D',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:30,verticalAlignment:'center'};
const audit=[];
for(const d of payload.tasks){
  if(chosen.size&&!chosen.has(d.task))continue;
  const wb=Workbook.create(),main=wb.worksheets.add('整体指标'),raw=wb.worksheets.add('逐测试样本');
  main.showGridLines=false;raw.showGridLines=false;main.tabColor='#116A9D';raw.tabColor='#DE9847';
  const maps=Object.fromEntries(['Sim','Exp'].map(s=>[s,new Map(d.stages[s].metrics.map(([g,k,v])=>[g+'\t'+k,v]))]));
  const allKeys=[...new Set([...d.required_metrics.map(k=>'overall\t'+k),...maps.Sim.keys(),...maps.Exp.keys()])];
  if(d.task==='lsp')allKeys.sort((a,b)=>(a.startsWith('PCK group/')?0:1)-(b.startsWith('PCK group/')?0:1));
  const values=allKeys.map(key=>{const [g,k]=key.split('\t');return [label(g),k,maps.Sim.get(key)??null,maps.Exp.get(key)??null,g.includes('counts')?'个':k==='epoch'?'轮':unit(k,d),g.includes('counts')?'—':dir(k)];});
  const meta=[];
  for(const s of ['Sim','Exp']){
    const r=d.stages[s];meta.push([s+' 核对状态',statusLabels[r.status]],
      [s+' 截图目标（非数据）',d.expected[s]],[s+' 归档主指标',r.primary_value??'最终版本未确认'],
      [s+' 逐样本记录条数',r.sample_rows],[s+' 有主指标的样本条数',r.sample_rows_with_primary_observation],
      [s+' 权重SHA256',r.checkpoint_sha256??'未定位'],
      [s+' 说明',r.notes],[s+' 缺失整体指标',r.missing_overall_metrics.join(', ')||'无']);
  }
  if(d.reference_sample_bindings){
    meta.push(['原截图Exp（最终身份待核）',d.original_screenshot_exp],['旧会话校正值（不作最终结果）',d.superseded_session_corrected_value]);
    for(const s of ['Sim','Exp'])meta.push([s+' 参考明细（不对应主列）',d.reference_sample_bindings[s].row_label+'：'+d.reference_sample_bindings[s].rows+'条']);
    meta.push(['Exp旧会话权重SHA（非最终）',d.stages.Exp.reference_checkpoint_sha256]);
  } else if(d.original_screenshot_exp!==undefined)meta.push(['原截图Exp（用户已更正）',d.original_screenshot_exp],['用户确认更正后的Exp目标',d.expected.Exp]);
  meta.push(['空白单元格',d.reference_sample_bindings?'指定Sim原排名未恢复；最终Exp身份未核，不用参考回放或已作废会话填充主列。参考明细保留，空白不是0。':'该阶段不适用或原记录未提供，不是0。单侧历史汇总不填另一侧；主指标及所需绘图明细已补齐。'],
    ['数据内容','原保存结果与固定权重CPU补算分开标注；无训练、无新光路采集，原件不改。'],
    ['原始证据 SHA256','以下每个文件都按本地现存内容计算SHA256；plot_data.json保留完整清单。']);
  for(let i=0;i<d.sources.length;i++){const p=d.sources[i];meta.push([`来源 ${i+1}`,p.path],[`来源 ${i+1} SHA256`,p.sha256]);}
  const end=8+values.length,maxRow=Math.max(end,8+meta.length);
  main.getRange(`A1:I${maxRow}`).format={font:{name:'Arial',size:10,color:'#203746'},rowHeight:24,verticalAlignment:'center'};
  for(const [i,w] of [30,45,20,20,16,18,3,27,92].entries())main.getRangeByIndexes(0,i,maxRow,1).format.columnWidth=w;
  main.getRange('A2:F2').merge();main.getRange('A2').values=[[d.name+'  |  LightGen-II']];main.getRange('A2:F2').format={font:{name:'Arial',size:17,bold:true,color:'#174D70'},rowHeight:36};
  main.getRange('A3:F3').merge();main.getRange('A3').values=[['Sim. 与括号内 Exp. 的归档核对　2026-10-26　｜　参照 T19 两表结构']];main.getRange('A3:F3').format={font:{name:'Arial',size:10,color:'#526B7B'},rowHeight:26};
  main.getRange('A4:F4').merge();main.getRange('A4').values=[[d.notes]];main.getRange('A4:F4').format={fill:'#FFF2D8',font:{name:'Arial',size:11,color:'#745319'},wrapText:true,rowHeight:80};
  main.getRange('A6:B6').merge();main.getRange('A6').values=[[d.reference_sample_bindings?'指定版本逐样本数（参考另列）':'逐样本条数（Sim. / Exp.）']];main.getRange('C6:D6').values=[[d.stages.Sim.sample_rows,d.stages.Exp.sample_rows]];main.getRange('C6:D6').setNumberFormat('#,##0');
  main.getRange('A8:F8').values=[['范围','指标','Sim.','Exp.','单位','方向']];main.getRange('A8:F8').format=header;
  main.getRange(`A9:F${end}`).values=values;main.getRange(`C9:D${end}`).setNumberFormat('0.000000');main.getRange(`C9:D${end}`).format.horizontalAlignment='right';
  for(let i=0;i<values.length;i++){
    const row=i+9;
    if(i%2===0)main.getRange(`A${row}:F${row}`).format.fill='#F0F5F8';
    if(values[i][4]==='个'||allKeys[i].includes('/per_joint_counts\t'))main.getRange(`C${row}:D${row}`).setNumberFormat('#,##0');
    if(values[i][0]==='整体'&&values[i][1]===d.primary)main.getRange(`A${row}:F${row}`).format={fill:'#DCECF5',font:{name:'Arial',size:11,bold:true,color:'#164D6D'}};
    for(const [j,column] of [[2,'C'],[3,'D']])if(values[i][j]===null)main.getRange(`${column}${row}`).format.fill=/historical|original_reported|reference/.test(allKeys[i])?'#F0F5F8':'#FCE5DF';
  }
  main.tables.add(`A8:F${end}`,true,'OverallMetrics');main.freezePanes.freezeRows(8);
  main.getRange('H8:I8').values=[['版本及证据','内容']];main.getRange('H8:I8').format=header;main.getRangeByIndexes(8,7,meta.length,2).values=meta;
  for(let i=0;i<meta.length;i++){
    const row=9+i,val=meta[i][1];if(i%2===0)main.getRange(`H${row}:I${row}`).format.fill='#F0F5F8';
    if(typeof val==='number')main.getRange(`I${row}`).setNumberFormat(Number.isInteger(val)?'#,##0':'0.0000000000');
    else{
      const length=[...String(val)].reduce((n,c)=>n+(c.charCodeAt(0)>255?2:1),0);
      main.getRange(`I${row}`).format.wrapText=true;main.getRange(`H${row}:I${row}`).format.rowHeight=Math.max(24,Math.ceil(length/87)*18+6);
    }
  }
  const n=d.samples.length,c=d.headers.length;
  raw.getRangeByIndexes(0,0,Math.max(n+1,4),c).format={font:{name:'Arial',size:10,color:'#203746'},rowHeight:23,verticalAlignment:'center'};
  raw.getRangeByIndexes(0,0,1,c).values=[d.headers];raw.getRangeByIndexes(0,0,1,c).format={...header,rowHeight:46,wrapText:true};
  if(n){raw.getRangeByIndexes(1,0,n,c).values=d.samples;raw.tables.add(`A1:${colName(c-1)}${n+1}`,true,'SavedTestSamples');}
  else {raw.getRange('A3:F3').merge();raw.getRange('A3').values=[['本版本逐查询原始排名尚未找回；此页不以其他版本或截图数值补造记录。']];raw.getRange('A3:F3').format={fill:'#FFF2D8',wrapText:true,rowHeight:55};}
  for(let i=0;i<c;i++){
    const key=d.headers[i],first=d.samples.find(r=>r[i]!==null)?.[i];
    const text=typeof first==='string';const w=i===0?(d.reference_sample_bindings?36:12):key==='sample_id'&&d.task.startsWith('lgvq')?80:/instruction|prompt/.test(key)?64:key==='positive_ranks'?64:/record_origin|record_status/.test(key)?36:/sample_id|query_id|source_id/.test(key)?32:typeof first==='string'?24:25;
    raw.getRangeByIndexes(0,i,Math.max(n+1,4),1).format.columnWidth=w;
    if(n&&key==='positive_ranks'){
      raw.getRangeByIndexes(1,i,n,1).format.wrapText=true;
      raw.getRangeByIndexes(1,i,n,1).format.rowHeight=d.task==='abo_text_to_image'?48:26;
    }
    if(n&&!text){raw.getRangeByIndexes(1,i,n,1).setNumberFormat(/rank|count|joints|_hit|^H@/.test(key)?'0':'0.000000');raw.getRangeByIndexes(1,i,n,1).format.horizontalAlignment='right';}
  }
  raw.freezePanes.freezeRows(1);raw.freezePanes.freezeColumns(2);
  wb.recalculate();
  if(!isDeepStrictEqual(main.getRange(`A9:F${end}`).values,values))throw Error('Summary readback mismatch '+d.task);
  if(n&&!isDeepStrictEqual(raw.getRangeByIndexes(1,0,n,c).values,d.samples))throw Error('Sample readback mismatch '+d.task);
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000});
  if(/"matchCount"\s*:\s*[1-9]/.test(errors.ndjson))throw Error(errors.ndjson);
  for(const [s,range,suffix] of [[main,`A2:I${maxRow}`,'overall'],[raw,`A1:${colName(Math.min(c,8)-1)}9`,'samples']]){
    const pic=await wb.render({sheetName:s.name,range,scale:1,format:'png'});await fs.writeFile(path.join(verify,d.task+'_'+suffix+'.png'),new Uint8Array(await pic.arrayBuffer()));
  }
  if(maxRow>32){const pic=await wb.render({sheetName:main.name,range:'A2:I32',scale:1,format:'png'});await fs.writeFile(path.join(verify,d.task+'_overall_top.png'),new Uint8Array(await pic.arrayBuffer()));}
  const expFirst=d.samples.findIndex(r=>r[0]===(d.reference_sample_bindings?.Exp.row_label??'Exp'))+2;
  for(const [range,suffix] of [[`E1:${colName(c-1)}9`,'metric_details'],[`A${expFirst}:${colName(Math.min(c,12)-1)}${expFirst+7}`,'samples_exp']]){
    const pic=await wb.render({sheetName:raw.name,range,scale:1,format:'png'});await fs.writeFile(path.join(verify,d.task+'_'+suffix+'.png'),new Uint8Array(await pic.arrayBuffer()));
  }
  const file=path.join(out,d.filename);await (await SpreadsheetFile.exportXlsx(wb)).save(file);
  const bytes=await fs.readFile(file);audit.push({task:d.task,filename:d.filename,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,sheets:2,source_sample_rows:n,full_in_memory_readback:true,error_scan:errors.ndjson,summary_values:values});
  console.log(JSON.stringify({task:d.task,file:d.filename,sample_rows:n,bytes:bytes.length}));
}
const merged=payload.tasks.map(d=>audit.find(x=>x.task===d.task)??previous.find(x=>x.task===d.task));
if(merged.some(x=>!x))throw Error('Missing prior workbook audit in scoped update');
await fs.writeFile(path.join(verify,'workbook_export_audit.json'),JSON.stringify({date:payload.date,workbooks:merged.length,files:merged},null,2));
console.log('Completed '+audit.length+' workbooks.');
