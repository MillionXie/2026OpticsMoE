/** Complete LG-system timing, not a pure-optical-time table. No forwards. */
import fs from 'node:fs/promises';import fss from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';import {fileURLToPath} from 'node:url';import {execFileSync} from 'node:child_process';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
import {stageChinese,familyChinese,baselineChinese,baselineDefinitions} from './lg_time_chinese_notes_20261025.mjs';
const HERE=path.dirname(fileURLToPath(import.meta.url)),REPO=path.resolve(HERE,'../../../..'),ANNOTATED=process.argv.includes('--annotated'),SEALED=path.join(HERE,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/complete_20261025_01'),OUT=ANNOTATED?path.join(path.dirname(SEALED),'timing_chinese_notes_20261025_01'):SEALED;
if(fss.existsSync(path.join(OUT,'delivery_manifest.json')))throw Error('Preserve sealed delivery');
const source=path.join(ANNOTATED?SEALED:OUT,'collected_data_and_audit.json'),data=JSON.parse(fss.readFileSync(source,'utf8')),sha=p=>crypto.createHash('sha256').update(fss.readFileSync(p)).digest('hex');
if(data.mainRecords.filter(x=>x.calls===600).length!==18)throw Error('All18 costs required');
const close=(a,b,label)=>{if(!Number.isFinite(a)||!Number.isFinite(b)||Math.abs(a-b)>Math.max(1e-6,Math.abs(b)*1e-9))throw Error(label+': '+a+' != '+b);};
const code=String.raw`
import csv,json,sys,math,collections
from pathlib import Path
R=Path(sys.argv[1]);d=json.loads(Path(sys.argv[2]).read_text(encoding='utf8'));csv.field_size_limit(10000000)
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def near(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6),(a,b)
out=[];seen=set()
for run in d['mainRuns']+d['tempRuns']:
 if run['folder'] in seen:continue
 seen.add(run['folder']);p=R/run['folder'];name=next(n for n in ('fields.csv','calls.csv','samples.csv') if (p/n).exists());calls=rows(p/name)
 start='input_start' if 'input_start' in calls[0] else 'preprocess_start';end='cpu_result_return' if 'cpu_result_return' in calls[0] else 'call_return'
 gaps=sum(float(calls[i][start])-float(calls[i-1][end]) for i in range(1,len(calls)))*1000;software=(float(calls[-1][end])-float(calls[0][start]))*1000
 stages=rows(p/'stages.csv') if (p/'stages.csv').exists() else [];spans=sorted((float(s['start_monotonic']),float(s['end_monotonic'])) for s in stages)
 assert all(b<=c+1e-8 for (a,b),(c,e) in zip(spans,spans[1:])),run['folder']
 bins={};parts=collections.Counter();cpu=None;cut=None;label=''
 for s in stages:
  wall=float(s['synchronized_wall_ms']);near(wall,(float(s['end_monotonic'])-float(s['start_monotonic']))*1000)
  region=s.get('region','');family=s['family'];charged=s['charged']=='True';key=(s['stage'],family,charged,region)
  g=bins.setdefault(key,dict(stage=s['stage'],family=family,charged=charged,region=region,ms=0.,events_ms=0.,event_count=0,calls=0));g['ms']+=wall;g['calls']+=1
  if s.get('cuda_event_ms') not in ('',None):g['events_ms']+=float(s['cuda_event_ms']);g['event_count']+=1
  if charged:
   part='input' if (region=='frontend' and family=='frontend_cpu' or family in ('input','input_transfer')) else 'front_nn' if (region=='frontend' or family in ('frontend','frontend_neural')) else 'body_output';parts[part]+=wall
 if run['system']=='ours':
  if name=='fields.csv':
   missing=sum(float(c['frontend_wall_ms']) for c in calls)-sum(g['ms'] for g in bins.values() if g['region']=='frontend');assert missing>=-1e-5;parts['unspanned_front']=max(0.,missing)
   cpu=sum(g['ms'] for g in bins.values() if g['region']=='frontend' and g['family']=='frontend_cpu')
   cut=sum(g['ms'] for g in bins.values() if g['stage'] in ('frontend.Qwen_decode65_448','frontend.raw100_224_decode'));label='视频解码 两路合计' if run['key'].startswith('01b') else '视频解码'
  else:
   candidate=max((g for g in bins.values() if g['charged'] and g['family']=='input'),key=lambda z:z['ms'],default=None);cut=candidate['ms'] if candidate else None;label=candidate['stage'] if candidate else '未独立打点'
  parts['gaps']=gaps;parts['fixed']=run['optical_s']*1000;near(sum(parts.values()),run['seconds200']*1000)
 else:
  parts=collections.Counter()
  if all(k in calls[0] for k in ('prepare_end','model_call_start','first_block_start','gpu_output')):
   for c in calls:
    v=[float(c[k]) for k in (start,'prepare_end','model_call_start','first_block_start','gpu_output',end)];assert all(b>=a for a,b in zip(v,v[1:]));
    for key,a,b in zip(('CPU输入准备','H2D','模型前端','主体网络和任务头','CPU输出'),v,v[1:]):parts[key]+=(b-a)*1000
   cpu=parts['CPU输入准备']
  elif all(k in calls[0] for k in ('prepare_end','model_start')):
   for c in calls:
    v=[float(c[k]) for k in (start,'prepare_end','model_start',end)];assert all(b>=a for a,b in zip(v,v[1:]));
    for key,a,b in zip(('CPU输入准备','H2D','模型 任务头 检索和输出'),v,v[1:]):parts[key]+=(b-a)*1000
   cpu=parts['CPU输入准备']
  elif stages:
   for g in bins.values():parts[g['stage']]+=g['ms']
   remainder=software-sum(parts.values())-gaps;assert remainder>=-1e-5;parts['未独立打点的软件段']=max(0.,remainder)
  else:parts['未独立拆分的完整调用']=software-gaps
  parts['请求间隙']=gaps;near(sum(parts.values()),software);near(software,run['seconds200']*1000)
 out.append(dict(key=run['key'],folder=run['folder'],system=run['system'],repeat=run['repeat'],valid=200,request_count=len(calls),gap_count=len(calls)-1,software_ms=software,gaps_ms=gaps,parts=dict(parts),stages=list(bins.values()),cpu_prepare_ms=cpu,cut_ms=cut,cut_label=label,calls_file=name))
print(json.dumps(out,ensure_ascii=True))
`;
const PYTHON='C:/Users/Xml12/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const runs=JSON.parse(execFileSync(PYTHON,['-c',code,REPO,source],{encoding:'utf8',maxBuffer:15*1024*1024}));if(runs.length!==90)throw Error('90 distinct leaves required');
const parallel=ANNOTATED?JSON.parse(execFileSync(PYTHON,[path.join(HERE,'lg_parallel_raw_notes_20261025.py'),REPO,source],{encoding:'utf8',maxBuffer:15*1024*1024})):null;
await fs.mkdir(OUT,{recursive:true});
const own=data.mainRecords.filter(x=>x.system==='ours'),base=data.mainRecords.filter(x=>x.system==='qwen'),group=k=>runs.filter(r=>r.key===k);
function agg(key){const rr=group(key);if(rr.length!==3)throw Error('Triple '+key);const parts={};for(const r of rr)for(const [k,v] of Object.entries(r.parts))parts[k]=(parts[k]||0)+v/600;return {parts,runs:rr,cut:rr.every(r=>r.cut_ms!==null)?rr.reduce((s,r)=>s+r.cut_ms,0)/600:null,cpu:rr.every(r=>r.cpu_prepare_ms!==null)?rr.reduce((s,r)=>s+r.cpu_prepare_ms,0)/600:null};}
const records=own.map(o=>{const q=base.find(x=>x.id===o.id),a=agg(o.key),b=agg(q.key);return {id:o.id,task:o.task,ours:o.ms,baseline:q.ms,parts:a.parts,cut_ms:a.cut,cut_label:a.runs[0].cut_label,LG_cpu_ms:a.cpu,base_cpu_ms:b.cpu};});
const details=[];for(const o of own){const bins=new Map();for(const r of group(o.key))for(const s of r.stages){const key=[s.stage,s.family,s.charged,s.region].join('|');if(!bins.has(key))bins.set(key,{...s,ms:0,events_ms:0,event_count:0,calls:0});const z=bins.get(key);for(const k of ['ms','events_ms','event_count','calls'])z[k]+=s[k];}for(const s of bins.values())details.push({id:o.id,task:o.task,...s,ms_per_input:s.ms/600,event_ms_per_input:s.event_count?s.events_ms/600:null});}
const baselineKeys=[...new Set(runs.filter(r=>r.system!=='ours').map(r=>r.key))],baselineParts=[];
for(const key of baselineKeys){const a=agg(key),r=data.mainRecords.find(x=>x.key===key)||data.tempRecords.find(x=>x.key===key);for(const [stage,ms] of Object.entries(a.parts))baselineParts.push({key,task:r.task,system:r.system,stage,ms});close(Object.values(a.parts).reduce((x,y)=>x+y,0),r.ms,'Baseline sum');}
const inputs=data.raw_files.filter(r=>/\/(stages\.csv|fields\.csv|calls\.csv|samples\.csv|summary\.json|report\.json)$/.test(r.file));for(const r of inputs){const p=path.join(REPO,r.file);if(fss.statSync(p).size!==r.bytes||sha(p)!==r.sha256)throw Error('Sealed raw differs '+r.file);}
const w=Workbook.create(),intro=ANNOTATED?w.worksheets.add('先读中文解释'):null,batchSheet=ANNOTATED?w.worksheets.add('视频并行与单帧'):null,s=w.worksheets.add('原主表时间拆分'),v=w.worksheets.add('扣除情景'),b=w.worksheets.add('Baseline时间拆分'),d=w.worksheets.add('LG逐阶段明细'),refs=w.worksheets.add('来源与口径');
const colors={ink:'#233B53',blue:'#E8F1FA',calc:'#EAF3EB',fixed:'#FFF1D6'},col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
function table(s,title,headers,values,widths){s.showGridLines=false;s.tabColor=colors.ink;s.getRange(`A1:${col(headers.length-1)}${values.length+8}`).format={font:{name:'Arial',size:11,color:colors.ink},verticalAlignment:'center'};s.getRange('A2').values=[[title]];s.getRange('A2').format.font={size:16,bold:true};s.getRange(`A5:${col(headers.length-1)}5`).values=[headers];s.getRange(`A5:${col(headers.length-1)}5`).format={fill:colors.ink,font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:56};s.getRange(`A6:${col(headers.length-1)}${values.length+5}`).values=values;s.getRange(`A6:${col(headers.length-1)}${values.length+5}`).format.rowHeight=32;widths.forEach((v,i)=>s.getRange(`${col(i)}5:${col(i)}${values.length+5}`).format.columnWidth=v);s.freezePanes.freezeRows(5);s.freezePanes.freezeColumns(1);}
const f=(s,c,v)=>s.getRange(c).formulas=[[v]],fmt=(s,r)=>{s.getRange(r).setNumberFormat('0.000');s.getRange(r).format.horizontalAlignment='right';};
table(s,'LG系统时间拆分 完整九任务',['任务及单位','LG 原主表\n总时间 ms','Baseline 原主表\n总时间 ms','原口径\n加速倍数','LG输入准备\n含合并H2D ms','LG前端NN\n及位置 ms','LG选定主体\n和输出 ms','真实请求\n间隙 ms','其余前端\n未细分 ms','固定执行\n阶段 ms','LG分项\n合计 ms'],records.map(r=>[r.task+(r.id.startsWith('01')?' 每视频':r.id==='04'?' 每标题':' 每样本'),r.ours,r.baseline,null,r.parts.input||0,r.parts.front_nn||0,r.parts.body_output||0,r.parts.gaps||0,r.parts.unspanned_front||0,r.parts.fixed,null]),[31,21,23,18,23,22,23,21,23,21,21]);
s.getRange('A3').values=[['先看B列：它与本次主表LG总时间完全相同。E至J列相加得到B列，非纯GPU时间。']];for(let i=0;i<9;i++){const n=i+6;f(s,'D'+n,`=C${n}/B${n}`);f(s,'K'+n,`=SUM(E${n}:J${n})`);}fmt(s,'B6:K14');s.getRange('B6:C14').format.fill=colors.blue;s.getRange('D6:D14').format.fill=colors.calc;s.getRange('J6:J14').format.fill=colors.fixed;
s.getRange('A17').values=[['E列沿原input标签分类，含解码、processor、tokenizer、合并H2D，部分任务也含embedding，不能整列当纯CPU或GPU。']];s.getRange('A18').values=[['前端NN及主体采用同步墙钟，包含区间内提交、同步或传输开销，并非仅CUDA Event。GPU FFT和批准排除项不重复加进总时间。']];
table(v,'去掉输入处理后倍率如何变化',['任务','LG原 ms','Baseline原 ms','拟从LG扣除的段','LG扣除 ms','LG剩余 ms','仅LG扣除\n倍率','LG已打点\nCPU准备 ms','Baseline CPU\n准备 ms','LG扣CPU后 ms','Baseline扣\nCPU后 ms','双方扣CPU\n倍率'],records.map(r=>[r.task,r.ours,r.baseline,r.cut_label,r.cut_ms??'未独立打点',null,null,r.LG_cpu_ms??'未独立打点',r.base_cpu_ms??'未独立打点',null,null,null]),[28,18,21,57,20,20,20,23,24,22,24,23]);
v.getRange('A3').values=[['G列为单侧扣除诊断，不是同口径加速。L列只扣双方各自已打点CPU准备，剩余也不是纯GPU。']];for(let i=0;i<9;i++){const n=i+6;f(v,'F'+n,`=IF(ISNUMBER(E${n}),B${n}-E${n},"未独立打点")`);f(v,'G'+n,`=IF(ISNUMBER(F${n}),C${n}/F${n},"未独立打点")`);f(v,'J'+n,`=IF(ISNUMBER(H${n}),B${n}-H${n},"未独立打点")`);f(v,'K'+n,`=IF(ISNUMBER(I${n}),C${n}-I${n},"未独立打点")`);f(v,'L'+n,`=IF(AND(ISNUMBER(J${n}),ISNUMBER(K${n})),K${n}/J${n},"未独立打点")`);}fmt(v,'B6:C14');fmt(v,'E6:L14');v.getRange('D6:D14').format={wrapText:true,rowHeight:60};
v.getRange('A17').values=[['LGVQ temporal扣296.017 ms视频解码；spatial扣279.696 ms和266.329 ms两路解码合计546.026 ms。']];v.getRange('A18').values=[['其他任务拟扣项为原input合并段，常同时含文件处理、processor、tokenizer和H2D。未拆分边界不填0。']];v.getRange('A19').values=[['情景是旧raw区间代数剔除，不是“已准备输入”新实测。主表及正式能耗不变；本表不虚构扣除后的新能耗。']];
table(b,'对照模型从原输入到CPU结果的计时',['配置键','任务','系统','计入阶段','三轮平均 ms/输入','计费状态',...(ANNOTATED?['具体中文说明','原阶段代码名']:[])],baselineParts.map(r=>[r.key,r.task,r.system,ANNOTATED?baselineChinese(r.stage)[0]:r.stage,r.ms,'包含在原总时间',...(ANNOTATED?[baselineChinese(r.stage)[1],r.stage]:[])]),[29,29,18,44,25,28,...(ANNOTATED?[95,60]:[])]);fmt(b,`E6:E${baselineParts.length+5}`);b.getRange('A3').values=[['含CPU读文件与解码、输入整理、送入显存、完整网络、任务输出和真实间隙。模型加载与独立离线图库另计。']];
table(d,'LG逐阶段中文说明与原始计时',['任务',ANNOTATED?'中文阶段名称':'原stage',ANNOTATED?'中文分类':'原family','所在部分','计入主表','同步总耗时\nms/视频或样本','显卡事件耗时\nms/视频或样本','三轮实际\n阶段调用数','计时性质',...(ANNOTATED?['这一段具体做什么','原阶段代码名','原分类代码名']:[])],details.map(r=>[r.task,ANNOTATED?stageChinese(r.stage)[0]:r.stage,ANNOTATED?familyChinese[r.family]:r.family,ANNOTATED?(r.region==='frontend'?'完整输入前端':r.region==='body'?'主体及输出':'原统一阶段'):r.region||'原统一阶段',r.charged?'计入':'实际执行但原约定排除',r.ms_per_input,r.event_ms_per_input??'未记录显卡事件',r.calls,r.family==='frontend_cpu'?'CPU上执行的输入处理':r.event_count?'包含等待完成的显卡相关区间，不是纯显卡算子时间':'CPU或未单独标记',...(ANNOTATED?[stageChinese(r.stage)[1],r.stage,r.family]:[])]),[27,46,29,22,28,25,27,22,57,...(ANNOTATED?[100,63,30]:[])]);fmt(d,`F6:G${details.length+5}`);d.getRange(`B6:C${details.length+5}`).format={wrapText:true,rowHeight:ANNOTATED?64:43};
if(ANNOTATED){
 b.getRange(`D6:D${baselineParts.length+5}`).format.wrapText=true;b.getRange(`G6:H${baselineParts.length+5}`).format={wrapText:true,rowHeight:60};
 d.getRange(`I6:L${details.length+5}`).format={wrapText:true,rowHeight:64};
 d.getRange('A3').values=[['先看B列中文名、J列具体说明。F列是计入边界内等待完成的总耗时，G列是显卡事件记录，不能相加。视频任务均按有效视频摊销。']];
 s.getRange('E5:G5').values=[['LG输入准备及\n合并传输 ms','LG前端神经网络\n及位置处理 ms','LG选定主体网络\n和结果输出 ms']];
 s.getRange('C5').values=[['对照模型原主表\n总时间 ms']];s.getRange('J5').values=[['固定光执行时间\n按参数计算 ms']];v.getRange('C5').values=[['对照模型原 ms']];
 s.getRange('A17').values=[['输入准备包含解码、图像整理、文字转词编号，合并段还含内存到显存传输，部分任务含词向量。不能整列称纯CPU。']];
 s.getRange('A18').values=[['网络计时为等待区间完成的总耗时，不只是显卡事件。仿真传播、布局、融合及原批准排除项保留在明细，未重复加进总时间。']];
 v.getRange('H5:L5').values=[['LG单独计时的\n全部CPU准备 ms','对照模型的\n全部CPU准备 ms','LG扣全部\nCPU准备后 ms','对照模型扣全部\nCPU准备后 ms','双方扣全部CPU\n准备后时间比']];
 for(let i=0;i<records.length;i++)v.getRange('D'+(i+6)).values=[[records[i].id.startsWith('01')?records[i].cut_label:stageChinese(records[i].cut_label)[0]]];
 v.getRange('A3').values=[['G列只从LG扣解码；L列从双方各自扣掉全部已单独计时的CPU准备。扣的项目不同，不能理解为仅再从对照模型扣时间。']];
 v.getRange('A18').values=[['其他任务拟扣项为原输入合并段，通常同时含文件处理、图像整理、文字分词及内存到显存传输。未拆分部分不填0。']];
 v.getRange('A19').values=[['扣除情景仅是原始区间的代数相减，不是用预先准备好的输入重新测速。正式时间与能耗不变。']];
 const definitionsRows=Object.entries(baselineDefinitions).slice(0,6).map(([alias,z])=>[z[0],alias,z[1]]);
 definitionsRows.push(
 ['计入CPU准备的内容','不是只有视频解码','LGVQ LG的CPU准备包括视频解码、图像整理、14维质量统计、文字分词；Qwen准备区间同样包含解码等，原记录未单拆其纯解码。'],
 ['为什么时间比会变大','Temporal实例','原比381.220/404.077=0.943。只扣LG解码是381.220/(404.077−296.017)=3.528。双方扣全部CPU准备是(381.220−295.399)/(404.077−385.878)=4.716。不是只扣Qwen。'],
 ['CPU','中央处理器及主内存','读文件、视频解码、文字分词等主要在CPU执行。CPU不是显卡。'],
 ['GPU','显卡及显存','承担张量网络计算。CPU内存的数据通常需先送入GPU显存。'],
 ['同步总耗时','原表的同步墙钟','开始计时，执行该段，再等待该段完成后停止。包含区间内提交、传输和同步开销。'],
 ['显卡事件耗时','CUDA Event','显卡事件记录的区间耗时，用于诊断。不含完整CPU输入流程，不能与同步总耗时重复相加。'],
 ['模型加载','LG与对照模型均不计入原推理时间','创建网络、从磁盘读取权重、将模型放入内存/显存发生在正式计时前。独立离线图库构建同样不计入在线查询。'],
 ['加载时间是否有统一统计','没有独立的纯加载耗时','已核90个合格点的身份、报告、状态及进程审计元数据，没有找到独立模型加载时长字段。加载期观察日志不等于纯加载秒数。'],
 ['如要计入加载','一次加载服务N个视频','一轮总时间=该轮加载时间+该轮N个视频总推理时间。平均每视频再除N，等效单帧再除4。三个独立新进程需计三次加载，不能只加一次。'],
 ['单帧是等效分摊','不表示只输入一帧也可完成评分','LGVQ每视频使用四帧，单帧等效=匹配总时间/有效帧数。双方同除4，加速比不变，不会凭换单位变成百倍。'],
 ['外层批数为1的含义','Temporal一个批含16视频','原身份中的batch=1指一次输入一个整场张量。Temporal图像特征形状[1,16,4,49,1024]，输出[1,16]。不是只跑一个视频。'],
 ['Temporal当前并行边界','主体整场64帧，输入前端未批量化','原前端逐视频解码，并逐视频调用图像分块/位置处理，随后一次把16视频64帧送入主体。不能宣称全流程64帧同时处理。'],
 ['Spatial当前配置限制','原模型只是一视频四帧整场','本次原Spatial输入[1,4,196,1024]，输出[1]。不符合16视频64帧配置，不能再直接除16。需要单独的批量/物理布局方案及新验证和新测量，不能靠改报表得到。'],
 ['LG硬件计费时间','原表称混合硬件投影','实际输入及选定电子网络耗时，加按参数计算的光执行时间。数值仿真传播确实执行，但不能再与固定光执行时间重复相加。这不是新光路整机实测。']);
 table(intro,'时间表中文解释',['中文名称或问题','原名称或结论','具体解释'],definitionsRows,[40,44,130]);intro.getRange(`A6:C${definitionsRows.length+5}`).format={wrapText:true,rowHeight:66};intro.getRange('A3').values=[['原始计时数字保持不变。先看本页解释，再看“视频并行与单帧”，代码名只为追溯保留。']];
 const batchRows=['01a|ours','01a|qwen','01b|ours','01b|qwen'].map(key=>{const rr=parallel.batch_runs.filter(x=>x.key===key);if(rr.length!==3)throw Error('Batch triple');const first=rr[0];return [key.startsWith('01a')?'时间质量':'空间质量',key.endsWith('ours')?'LG':'Qwen',rr.reduce((a,r)=>a+r.fields,0),first.videos_per_field,first.frames_per_field,600,rr.reduce((a,r)=>a+r.executed_slots,0),rr.reduce((a,r)=>a+r.total_ms,0),null,null,null,null,key==='01a|ours'?'主体16视频64帧整场；输入前端逐视频。每轮末场8有效+8补齐。':key==='01b|ours'?'原模型仅1视频4帧，不能称16视频并行，也不能再除16。':'单视频四帧一次调用，按4帧折算等效单帧。'];});
 table(batchSheet,'实际批量及等效单帧计算',['任务','系统','三轮整场\n或调用数','每场\n视频数','每场\n帧数','三轮有效\n视频数','三轮执行\n视频槽数','三轮总\n时间 ms','平均整场\n时间 ms','每有效视频\n时间 ms','每有效帧\n时间 ms','每执行帧\n摊销 ms','实际并行边界'],batchRows,[22,15,20,18,18,22,23,24,24,24,24,24,100]);
 for(let i=0;i<4;i++){const n=i+6;f(batchSheet,'I'+n,`=H${n}/C${n}`);f(batchSheet,'J'+n,`=H${n}/F${n}`);f(batchSheet,'K'+n,`=J${n}/4`);f(batchSheet,'L'+n,`=H${n}/(G${n}*4)`);}fmt(batchSheet,'H6:L9');batchSheet.getRange('M6:M9').format={wrapText:true,rowHeight:66};
 batchSheet.getRange('A3').values=[['Temporal：每轮13场×16视频×4帧，208执行槽但200有效视频。有效单帧用200×4作分母，补齐成本不删除。']];
 batchSheet.getRange('A11').values=[['Temporal平均整场时间÷64得到每执行帧摊销；正式按有效帧分摊还承担4%的补齐成本。双方同换成有效单帧后，时间比保持0.943倍。']];
 const temporalFields=parallel.fields.filter(r=>r.key==='01a|ours');
 batchSheet.getRange('A14').values=[['Temporal原始39个整场逐场记录（每场主体均16视频64帧）']];
 batchSheet.getRange('A16:K16').values=[['轮次','场序从0起','视频槽数','帧数','有效视频数','补齐视频数','输入前端 ms','选定主体及输出 ms','场间真实空档 ms','固定光执行 ms','本场LG计费总时间 ms']];
 batchSheet.getRange('A16:K16').format={fill:colors.ink,font:{color:'#FFFFFF',bold:true},wrapText:true,rowHeight:52};
 batchSheet.getRange(`A17:K${16+temporalFields.length}`).values=temporalFields.map(r=>[r.repeat,r.field,r.slots,r.frames,r.valid,r.padding,r.frontend_ms,r.body_ms,r.gap_ms,r.fixed_ms,r.total_ms]);fmt(batchSheet,`G17:K${16+temporalFields.length}`);
 await fs.writeFile(path.join(OUT,'parallel_and_loading_evidence.json'),JSON.stringify(parallel,null,2));
}
const definitions=[['原主表时间','本次首200×3 fresh口径总时间，不是历史窄口径或GPU Event合计。'],['LG计费','实际输入准备及选定电子同步墙钟，加固定执行阶段和真实请求间隙。LGVQ完整前端只算一次。'],['双侧CPU剔除','LGVQ LG扣frontend_cpu各段，baseline扣prepare_end−preprocess_start。未细分CPU调度可能仍留在剩余。'],['单侧剔除','baseline原总时间/(LG原总时间−指定输入段)。只用于诊断该段影响。'],['GPU时间','同步墙钟和CUDA Event分列。没有重测纯GPU模式，不以Event替换正式主表。'],['能耗','正式能耗仍为原完整计费窗；扣除情景未重新积分能量。'],['LGVQ分母','temporal13整场含208槽，200有效视频；spatial200整场。每视频4帧，等效帧除4。'],['汇总来源 SHA256',sha(source)],...inputs.map(r=>[r.file,`${r.bytes} bytes SHA256 ${r.sha256}`])];
table(refs,'定义与原始时间文件索引',['项目或文件','定义或SHA256'],definitions,[103,125]);refs.getRange(`A6:B${definitions.length+5}`).format={wrapText:true,rowHeight:58};
if(ANNOTATED){refs.getRange('A6:B12').values=[
 ['原主表时间','首200输入、三个独立新进程的完整计费时间，不是历史窄范围计时或显卡事件合计。'],
 ['LG计费','实际输入及选定电子网络同步总耗时，加参数光执行时间和真实请求间隙。完整输入前端只算一次。'],
 ['双方扣CPU准备','LGVQ LG扣掉已单独计时的全部CPU准备。对照模型扣掉开始准备到准备完成的整个CPU区间。剩余仍可能含未拆出的CPU调度。'],
 ['只从LG扣除','对照模型原总时间除以LG原总时间减指定输入段，只用于看该段影响，不是相同计费范围的加速比。'],
 ['显卡相关时间','同步总耗时与显卡事件分列，不能相加。没有用显卡事件替换正式主表的完整输入计时。'],
 ['能耗','正式能耗仍对应原完整计费范围。扣时间的分析情景没有重新测量能耗，不推算虚假的扣除后能耗。'],
 ['LGVQ分母与并行边界','Temporal每场16视频64帧，13场含208执行槽、200有效视频；Spatial原配置一视频四帧。双方每有效视频时间除4得到等效单帧。']];}
w.recalculate();for(let i=0;i<9;i++){const n=i+6;close(s.getRange('K'+n).values[0][0],records[i].ours,'LG decomposition');close(s.getRange('D'+n).values[0][0],records[i].baseline/records[i].ours,'Original ratio');}
const first=s.getRange('K6').values[0][0],input=s.getRange('E6').values[0][0];s.getRange('E6').values=[[input+1]];close(s.getRange('K6').values[0][0],first+1,'Live sum');s.getRange('E6').values=[[input]];w.recalculate();
await fs.writeFile(path.join(OUT,'lg_time_breakdown.json'),JSON.stringify({status:'COMPLETE_9_LG_AND_ALL_BASELINE_TIMING_BREAKDOWN',records,runs,details,baseline_parts:baselineParts,no_models_or_GPU_calls:true,scenario_is_subtraction_not_new_measurement:true,source_sha256:sha(source)},null,2));
const scan=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});await fs.writeFile(path.join(OUT,'LG_time_formula_scan.ndjson'),scan.ndjson);
for(const [sheet,range,tag] of [[s,'A1:F18','primary'],[s,'G4:K14','parts'],[v,'A1:G19','scenario'],[v,'H4:L14','both_CPU'],[b,'A1:F17','baselines'],[d,'A4:I14','stages'],[refs,'A4:B13','definitions'],...(ANNOTATED?[[intro,'A1:C13','plain_explanations'],[intro,'A18:C24','loading_notes'],[batchSheet,'A1:G11','parallel_counts'],[batchSheet,'H4:M11','frame_numbers'],[batchSheet,'A14:K30','field_records'],[b,'D4:H12','baseline_notes'],[d,'I4:L13','stage_notes']]:[])]){const blob=await w.render({sheetName:sheet.name,range,scale:1.25,format:'png'});await fs.writeFile(path.join(OUT,'LG_time_'+tag+'.png'),new Uint8Array(await blob.arrayBuffer()));}
const book=await SpreadsheetFile.exportXlsx(w);const filename=ANNOTATED?'A100_LG时间拆分_中文注释与64帧核对_20261025.xlsx':'A100_LG系统时间拆分与扣除对照_20261025.xlsx';await book.save(path.join(OUT,filename));console.log(JSON.stringify({tasks:9,unique_points:runs.length,LG_stage_groups:details.length,baseline_configurations:baselineKeys.length,filename}));
