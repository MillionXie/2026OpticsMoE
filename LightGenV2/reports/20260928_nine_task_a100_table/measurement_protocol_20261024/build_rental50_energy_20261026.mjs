import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(here,'../../../..');
const root=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/a10080_energy_triples_20261026_01');
const out=path.join(here,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/rental50_energy_20261026_01');
const target=path.join(out,'A10080_Temporal_MPS50_时间能耗与阶段明细_20261026.xlsx');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const sha=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
const close=(a,b)=>{if(Math.abs(a-b)>Math.max(1e-5,Math.abs(b)*1e-9))throw Error(`Mismatch ${a} ${b}`);};
try{await fs.access(path.join(out,'delivery_manifest.json'));throw Error('Preserve sealed delivery');}catch(e){if(e.code!=='ENOENT')throw e;}
const points=[],files=[];
for(const [model,repeats] of [['kimi',3],['deepseek',2]])for(let repeat=1;repeat<=repeats;repeat++){
  const leaf=path.join(root,`${model}50_repeat0${repeat}`),cost=await read(path.join(leaf,'cost_report.json'));
  const audit=await read(path.join(leaf,'local_independent_energy_audit01.json'));
  const receipt=await read(path.join(leaf,'backup_receipt_complete01.json'));
  if(cost.status!=='complete'||audit.status!=='passed'||cost.cap!==50||cost.video_calls!==2048||!audit.four_cycles_bitwise_equal)throw Error('Incomplete original evidence');
  for(const x of receipt.verified_files){
    const local=path.join(leaf,x.path);if((await fs.stat(local)).size!==x.bytes||await sha(local)!==x.sha256)throw Error(`Original raw changed ${local}`);
    files.push({model,repeat,kind:'远端原文件，封存SHA一致',local,remote:receipt.source+'/'+x.path,bytes:x.bytes,sha256:x.sha256});
  }
  for(const [name,digest] of Object.entries(audit.raw_sha256))if(await sha(path.join(leaf,name))!==digest)throw Error('Recorded audit source mismatch');
  for(const name of ['backup_receipt_complete01.json','local_independent_energy_audit01.json']){const local=path.join(leaf,name);files.push({model,repeat,kind:'本地备份收据或独立审计',local,remote:'不适用，本地生成',bytes:(await fs.stat(local)).size,sha256:await sha(local)});}
  close(cost.time16_ms,audit.time16_ms);close(cost.energy16_j,audit.energy16_j);
  close(cost.elapsed_s*1000/128,cost.time16_ms);close(cost.energy_j/128,cost.energy16_j);
  close(cost.common_Qwen_effective_tokens/cost.elapsed_s,cost.common_tokens_per_s);
  points.push({model,repeat,leaf,cost,audit,cost_sha256:await sha(path.join(leaf,'cost_report.json')),audit_sha256:await sha(path.join(leaf,'local_independent_energy_audit01.json')),receipt_sha256:await sha(path.join(leaf,'backup_receipt_complete01.json'))});
}
const result={status:'passed',scope:'new corrected serial accounting, MPS50, Kimi 3 rounds and DeepSeek 2 rounds',platform:'A100-SXM4-80GB / 400W / 54 visible SM',calls_per_round:2048,unique_videos_per_round:512,cycles:4,frames_per_video:4,points:points.map(({cost,...p})=>({...p,cost:{...cost,cost_intervals:undefined}})),files};
await fs.mkdir(out,{recursive:true});
await fs.writeFile(path.join(out,'collected_results.json'),JSON.stringify(result,null,2));
const csvCell=x=>'"'+String(x).replaceAll('"','""')+'"';
await fs.writeFile(path.join(out,'raw_file_index.csv'),'\uFEFF'+[['model','repeat','kind','local','remote','bytes','sha256'],...files.map(x=>Object.values(x))].map(r=>r.map(csvCell).join(',')).join('\r\n'));

const w=Workbook.create(),summary=w.worksheets.add('结果与口径'),rounds=w.worksheets.add('轮次原值');
for(const [s,rows,cols] of [[summary,40,8],[rounds,6,33]]){s.showGridLines=false;s.getRangeByIndexes(0,0,rows,cols).format={font:{name:'Arial',size:11,color:'#182B3A'},rowHeight:26,verticalAlignment:'center'};}
summary.tabColor='#243B53';
const put=(s,r,rows)=>s.getRangeByIndexes(r-1,0,rows.length,rows[0].length).values=rows;
const head=(s,r,cols)=>s.getRangeByIndexes(r-1,0,1,cols).format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:52};
const cols=['模型','轮次','视频调用数','保留累计时间/s','同区间能量/J','共同Qwen有效tokens','自身有效tokens','每16视频/ms','等效单帧/ms','每16视频/J','等效单帧/J','共同tokens/s','自身tokens/s','视频/s','等效帧/s','GPU积分能量/J','GPU时间加权功率/W','主机代理/W','CPU文字分词/ms每16','完整GPU模型至评分/ms每16','模型入口调度/ms每16','中间CPU返回/ms每16','视频间调度/ms每16','已排除图像准备/ms每16','已排除原始搬运/ms每16','末条CPU返回，窗外/ms每16','最大功率样本年龄/s','资格','原cost绝对路径','原cost SHA256','本地audit SHA256','评分NPZ SHA256','备份收据 SHA256'];
put(rounds,1,[cols]);head(rounds,1,cols.length);
points.forEach((p,i)=>{const c=p.cost,a=p.audit,s=a.stage_equivalent16_ms,r=i+2;
  put(rounds,r,[[p.model==='kimi'?'Kimi':'DeepSeek完整',p.repeat,c.video_calls,c.elapsed_s,c.energy_j,c.common_Qwen_effective_tokens,c.native_effective_tokens,null,null,null,null,null,null,null,null,c.GPU_energy_j,c.observed_timeweighted_GPU_w,c.host_proxy_w,s.CPU_tokenizer_retained_ms,s.full_GPU_model_to_score_host_ms,s.entry_dispatch_retained_ms,s.intermediate_CPU_output_retained_ms,s.intervideo_dispatch_retained_ms,s.CPU_image_preparation_excluded_ms,s.H2D_excluded_ms,s.final_CPU_output_outside_primary_ms,a.max_power_sample_age_s,'合格且本地原SHA备份',path.join(p.leaf,'cost_report.json'),p.cost_sha256,p.audit_sha256,a.raw_sha256['outputs.npz'],p.receipt_sha256]]);
  rounds.getRange(`H${r}:O${r}`).formulas=[[`=D${r}*1000/C${r}*16`,`=H${r}/64`,`=E${r}/C${r}*16`,`=J${r}/64`,`=F${r}/D${r}`,`=G${r}/D${r}`,`=C${r}/D${r}`,`=N${r}*4`]];
});
rounds.freezePanes.freezeRows(1);rounds.freezePanes.freezeColumns(2);
rounds.getRange('A:A').format.columnWidth=23;rounds.getRange('B:AB').format.columnWidth=19;rounds.getRange('AC:AG').format.columnWidth=75;
rounds.getRange('C2:C6').setNumberFormat('#,##0');rounds.getRange('D2:AA6').setNumberFormat('0.000');rounds.getRange('F2:G6').setNumberFormat('#,##0');rounds.getRange('AA2:AA6').setNumberFormat('0.000000');
summary.getRange('A:A').format.columnWidth=37;summary.getRange('B:H').format.columnWidth=22;
put(summary,2,[['A100 80GB：Temporal，MPS50%已保全数据']]);summary.getRange('A2').format.font={size:16,bold:true};
put(summary,3,[['Kimi三轮；DeepSeek两轮。每轮2048次＝512个原TEST视频循环4遍，每视频4帧。']]);
put(summary,4,[['A100-SXM4-80GB，原400W上限、实际54SM；严格逐视频串行，无预取、跨视频并行或GPU指令图。']]);
put(summary,6,[['模型','合格轮数','视频调用合计','每16视频/ms','等效单帧/ms','每16视频/J','共同tokens/s','自身tokens/s']]);head(summary,6,8);
for(const [r,label,lo,hi] of [[7,'Kimi',2,4],[8,'DeepSeek完整',5,6]]){
  put(summary,r,[[label,hi-lo+1,null,null,null,null,null,null]]);
  summary.getRange(`C${r}:H${r}`).formulas=[[`=SUM('轮次原值'!C${lo}:C${hi})`,`=SUM('轮次原值'!D${lo}:D${hi})*1000/C${r}*16`,`=D${r}/64`,`=SUM('轮次原值'!E${lo}:E${hi})/C${r}*16`,`=SUM('轮次原值'!F${lo}:F${hi})/SUM('轮次原值'!D${lo}:D${hi})`,`=SUM('轮次原值'!G${lo}:G${hi})/SUM('轮次原值'!D${lo}:D${hi})`]];
}
summary.getRange('C7:C8').setNumberFormat('#,##0');summary.getRange('D7:H8').setNumberFormat('#,##0.000');
put(summary,10,[['tokens/s均为总有效token数÷总保留秒数，不平均各轮速率。共同分子每轮913408；自身token定义因模型不同不能互换。']]);
put(summary,12,[['逐阶段时间（ms/16视频）','Kimi三轮合并','DeepSeek两轮合并','是否计入主时间','中文解释']]);head(summary,12,5);
const stages=[['文字分词','S','计入','CPU把文字切为词编号，不是词向量查表。'],['完整GPU模型至评分','T','计入','原GPU图像/文字embedding、全部网络、完整词表和任务头，以及模型内真实等待。'],['模型入口调度','U','计入','原始搬运结束到模型计时入口之间的CPU下发和调度。'],['中间结果回到CPU','V','计入','前2047个视频评分就绪后，结果返回CPU的等待。'],['视频之间的调度','W','计入','上一视频返回后到下一视频准备开始之间的真实间隙。'],['保留时间合计',null,'主结果','以上五项之和。'],['视频解码、取帧、裁剪和图像整理','X','排除','原始操作逐视频实际执行，只在核算中逐条扣除。'],['原始CPU到GPU搬运','Y','排除','主机内存到显存，简称H2D；不包含模型中间传输。'],['最后一个视频结果回到CPU','Z','窗外另列','主终点为最后GPU评分就绪，末条CPU返回另列。']];
for(let i=0;i<stages.length;i++){const [label,col,scope,note]=stages[i],r=13+i;put(summary,r,[[label,null,null,scope,note]]);summary.getRange(`B${r}:C${r}`).formulas=col?[[`=SUM('轮次原值'!${col}2:${col}4)/$B$7`,`=SUM('轮次原值'!${col}5:${col}6)/$B$8`]]:[['=SUM(B13:B17)','=SUM(C13:C17)']];}
summary.getRange('B13:C21').setNumberFormat('0.000');summary.getRange('A13:D21').format.rowHeight=32;summary.getRange('E13:E21').format={columnWidth:76,wrapText:true,rowHeight:44};
summary.getRange('D13:D21').format.horizontalAlignment='center';summary.getRange('B21:C21').setNumberFormat('0.000000');
put(summary,24,[['能量：对同一组保留时间区间积分原50ms GPU功率，再加338.2W主机功率代理×保留秒数。不扣空闲功率，不含被排除准备区间。']]);
put(summary,25,[['这是GPU观察＋主机代理估算，不是插座电表整机实测。保留慢轮；未测轮不补零、不借旧口径能耗。']]);
put(summary,26,[['本表不混入40GB结果或LG理想覆盖估算。正式准确度仍沿用原完整TEST，不以循环512视频替代。']]);
put(summary,28,[['样本轮间范围','最小ms/16','最大ms/16','最小J/16','最大J/16']]);head(summary,28,5);
summary.getRange('A29:A30').values=[['Kimi（三轮）'],['DeepSeek（两轮）']];
summary.getRange('B29:E30').formulas=[["=MIN('轮次原值'!H2:H4)","=MAX('轮次原值'!H2:H4)","=MIN('轮次原值'!J2:J4)","=MAX('轮次原值'!J2:J4)"],["=MIN('轮次原值'!H5:H6)","=MAX('轮次原值'!H5:H6)","=MIN('轮次原值'!J5:J6)","=MAX('轮次原值'!J5:J6)"]];summary.getRange('B29:E30').setNumberFormat('0.000');
w.recalculate();
for(const [r,model,stage] of [[7,'kimi','B18'],[8,'deepseek','C18']]){const ps=points.filter(p=>p.model===model),total=ps.reduce((a,p)=>a+p.cost.elapsed_s,0),n=ps.length*2048;close(summary.getRange(`D${r}`).values[0][0],total*1000/n*16);close(summary.getRange(`G${r}`).values[0][0],ps.length*913408/total);close(summary.getRange(`D${r}`).values[0][0],summary.getRange(stage).values[0][0]);}
// Recalculation proof: change a real source, observe a dependent result, restore.
const original=rounds.getRange('D2').values[0][0],before=summary.getRange('D7').values[0][0];rounds.getRange('D2').values=[[original+1]];w.recalculate();close(summary.getRange('D7').values[0][0]-before,1000/6144*16);rounds.getRange('D2').values=[[original]];w.recalculate();close(summary.getRange('D7').values[0][0],before);
console.log((await w.inspect({kind:'table',range:'结果与口径!A6:H8',include:'values,formulas',tableMaxRows:3,tableMaxCols:8,maxChars:3500})).ndjson);
console.log((await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson);
await (await SpreadsheetFile.exportXlsx(w)).save(target);
for(const [sheetName,range,name] of [['结果与口径','A2:H10','summary'],['结果与口径','A12:E21','stages'],['轮次原值','A1:M6','rounds']]){const b=await w.render({sheetName,range,scale:1.5,format:'png'});await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await b.arrayBuffer()));}
await fs.writeFile(path.join(out,'builder_checks.json'),JSON.stringify({status:'passed',new_GPU_or_model_calls:0,remote_raw_files:files.filter(x=>x.kind.startsWith('远端')).length,files:files.length,recalculation_input_mutation_restored:true,worksheet_stage_sum_matches_primary:true,qualified_rounds:{kimi:3,deepseek:2},workbook_sha256:await sha(target)},null,2));
console.log(JSON.stringify({out,target,sha256:await sha(target),bytes:(await fs.stat(target)).size,verified_original_files:55}));
