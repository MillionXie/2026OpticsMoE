/** File-only optical/electronic timing decomposition. No model or GPU work. */
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const HERE=path.dirname(fileURLToPath(import.meta.url)),REPO=path.resolve(HERE,'../../../..');
const ROOT=path.join(HERE,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83');
const PRIOR=path.join(ROOT,'collected_20261025_01'),OUT=path.join(ROOT,'optical_breakdown_20261025_01');
const PYTHON='C:/Users/Xml12/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const norm=p=>p.replaceAll('\\','/'),rel=p=>norm(path.relative(REPO,p));
const read=p=>JSON.parse(fss.readFileSync(p,'utf8'));
const digest=p=>crypto.createHash('sha256').update(fss.readFileSync(p)).digest('hex');
function assert(x,label){if(!x)throw Error(label);}
function close(a,b,label){assert(Math.abs(a-b)<=Math.max(1e-6,Math.abs(b)*1e-9),`${label}: ${a} != ${b}`);}
const previous=read(path.join(PRIOR,'delivery_manifest.json'));
assert(digest(path.join(PRIOR,'delivery_manifest.json'))==='5e39e2122061290b057b46fbb7eb70b0ec8de37ea726f8bdd54925810a75f236','Prior delivery manifest identity differs');
const source=path.join(PRIOR,'collected_data_and_audit.json');
assert(digest(source)===previous.outputs.find(x=>x.file==='collected_data_and_audit.json').sha256,'Collected source identity differs');
const data=read(source),protocol=read(path.join(HERE,'protocol.json')),opt=protocol.lightgen.optical;
close(opt.slm_ms+opt.exposure_ms+opt.additional_ccd_ms,opt.per_pass_ms,'Single optical cycle');
assert(opt.slm_ms===.714&&opt.exposure_ms===.300&&opt.additional_ccd_ms===.0307,'Fixed approved timing parameters differ');
const runs=data.mainRuns.filter(x=>x.system==='ours'),pinMap=new Map(data.raw_files.map(x=>[x.file,x]));
const refs=[];
function reference(p,kind){const abs=path.isAbsolute(p)?p:path.join(REPO,p);const record={path:rel(abs),bytes:fss.statSync(abs).size,sha256:digest(abs),kind};refs.push(record);return record;}
for(const r of runs)for(const name of [r.reportName,'stages.csv','stage_ledger.json',fss.existsSync(path.join(REPO,r.folder,'calls.csv'))?'calls.csv':'samples.csv']){
 const p=r.folder+'/'+name,pin=pinMap.get(p);assert(pin,`Missing original pin ${p}`);
 assert(pin.verification==='封存SHA及字节数已核对',`Original sealed pin required ${p}`);
 const current=reference(p,'原始计量');assert(current.sha256===pin.sha256&&current.bytes===pin.bytes,`Raw identity differs ${p}`);
}
for(const name of ['protocol.json','measure_lgvq_cost.py','lgvq_live_frontend.py','measure_t12_cost.py','measure_ours.py','measure_abo_title_student.py','measure_abo_image_student.py','measure_lsp.py','measure_salicon.py','measure_openmoji_e45.py'])reference(path.join(HERE,name),'协议或计算图源码');
reference(path.join(REPO,'LightGenV2/tasks/t06_video_quality_assessment/lab_runtime.py'),'LGVQ六光路身份');
reference(source,'已交付计量汇总');reference(path.join(PRIOR,'delivery_manifest.json'),'已交付身份清单');
// Python CSV is used only for reading/aggregating original observations.
const extract=String.raw`
import csv,json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);data=json.loads(Path(sys.argv[2]).read_text(encoding='utf8'));result=[]
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
for run in data['mainRuns']:
 if run['system']!='ours':continue
 folder=root/run['folder'];stages=rows(folder/'stages.csv');calls=rows(folder/('calls.csv' if (folder/'calls.csv').exists() else 'samples.csv'))
 assert len(calls)==200
 start='input_start' if 'input_start' in calls[0] else 'preprocess_start';end='cpu_result_return' if 'cpu_result_return' in calls[0] else 'call_return'
 gaps=sum(float(calls[i][start])-float(calls[i-1][end]) for i in range(1,200))*1000
 software=(float(calls[-1][end])-float(calls[0][start]))*1000
 intervals=sorted((float(s['start_monotonic']),float(s['end_monotonic'])) for s in stages)
 assert all(b<=c+1e-8 for (a,b),(c,d) in zip(intervals,intervals[1:])),run['folder']
 groups={}
 for s in stages:
  key=(s['stage'],s['family'],s['charged']=='True');g=groups.setdefault(key,dict(stage=key[0],family=key[1],charged=key[2],calls=0,ms=0.))
  g['calls']+=1;g['ms']+=float(s['synchronized_wall_ms'])
  assert math.isclose(float(s['synchronized_wall_ms']),(float(s['end_monotonic'])-float(s['start_monotonic']))*1000,rel_tol=1e-8,abs_tol=1e-7)
 charged=sum(g['ms'] for g in groups.values() if g['charged'])+gaps
 assert math.isclose(charged,run['charged_s']*1000,rel_tol=1e-9,abs_tol=1e-6)
 remainder=software-sum(g['ms'] for g in groups.values())-gaps;assert remainder>=-1e-5
 result.append(dict(key=run['key'].split('|')[0],folder=run['folder'],software_ms=software,gap_ms=gaps,remainder_ms=remainder,stages=list(groups.values())))
print(json.dumps(result,ensure_ascii=False))
`;
const extracted=JSON.parse(execFileSync(PYTHON,['-c',extract,REPO,source],{encoding:'utf8',maxBuffer:8*1024*1024}));
assert(extracted.length===18,'Exactly original six Ours triples required');
const aggregate=new Map(),summary=new Map();
function add(key,stage,family,charged,calls,ms){const k=[key,stage,family,charged].join('|');if(!aggregate.has(k))aggregate.set(k,{key,stage,family,charged,calls:0,ms:0});const g=aggregate.get(k);g.calls+=calls;g.ms+=ms;}
for(const run of extracted){
 const s=summary.get(run.key)||{software_ms:0,original_runs:0};s.software_ms+=run.software_ms;s.original_runs++;summary.set(run.key,s);
 for(const g of run.stages)add(run.key,g.stage,g.family,g.charged,g.calls,g.ms);
 add(run.key,'__inter_request_gap','request_gap',true,199,run.gap_ms);
 add(run.key,'__unspanned_software_overhead','software_remainder',false,200,run.remainder_ms);
}
const records=data.mainRecords.filter(x=>x.system==='ours').map(r=>({...r,id:r.key.split('|')[0]}));
const stages=[...aggregate.values()].sort((a,b)=>a.key.localeCompare(b.key)||Number(b.charged)-Number(a.charged)||a.family.localeCompare(b.family)||a.stage.localeCompare(b.stage));
for(const r of records.filter(x=>x.calls)){
 const g=stages.filter(x=>x.key===r.id),own=runs.filter(x=>x.key===r.key);
 close(g.filter(x=>x.charged).reduce((s,x)=>s+x.ms,0),own.reduce((s,x)=>s+x.charged_s*1000,0),'Electronic charged ledger');
 close(g.reduce((s,x)=>s+x.ms,0),summary.get(r.id).software_ms,'Software full-window reconciliation');
 close(own.reduce((s,x)=>s+x.optical_s*1000,0)/600,['04','05','06'].includes(r.id)?3.1341:6.2682,'Original optical cycle total');
}
const descriptions={
 input:'文件、文本、解码、预处理或H2D',frontend:'Embedding、patch或位置准备',bridge:'真实merger或跨模态连接',adapter:'输入/输出adapter与norm',
 amplitude_encoding_nn:'电子幅度编码和归一化',router:'电子路由能量、top-k与权重',ccd_nonlinearity:'电子CCD值归一化/非线性',ccd_readout:'电子CCD池化与可学习读出',
 head:'原任务头或decoder',search:'原检索和排序',output:'CPU结果与输出后处理',request_gap:'199个真实请求间隙/轮',simulation_only:'GPU FFT数值传播，诊断不计费',parallel_residual:'并行电子残差，执行但不计费',layout:'软件场布局，执行但不计费',fusion:'软件融合，执行但不计费',software_remainder:'原软件全窗减已打点区间和请求间隙，未细分软件开销'};
const names=Object.fromEntries(records.map(r=>[r.id,r.task]));
const w=Workbook.create(),s=w.worksheets.add('任务时间拆分'),o=w.worksheets.add('光路逐步骤'),d=w.worksheets.add('电子与仿真明细'),refsSheet=w.worksheets.add('原始来源');
const colors={navy:'#233B53',ink:'#203447',blue:'#E8F1FA',green:'#E9F3EB',amber:'#FFF0D8',grey:'#F2F3F5'};
for(const sheet of [s,o,d,refsSheet]){sheet.showGridLines=false;sheet.getRange('A1:N420').format.font={name:'Arial',size:11,color:colors.ink};sheet.tabColor=colors.navy;}
const col=i=>{let x='';for(i++;i;i=Math.floor((i-1)/26))x=String.fromCharCode(65+(i-1)%26)+x;return x;};
function heading(sheet,at,text){sheet.getRange(at).values=[[text]];sheet.getRange(at).format.font={size:15,bold:true};}
function table(sheet,row,headers,rows,widths){const last=col(headers.length-1);sheet.getRange(`A${row}:${last}${row}`).values=[headers];sheet.getRange(`A${row}:${last}${row}`).format={fill:colors.navy,font:{bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:46};if(rows.length)sheet.getRange(`A${row+1}:${last}${row+rows.length}`).values=rows;
 sheet.getRange(`A${row+1}:${last}${row+rows.length}`).format={rowHeight:30,verticalAlignment:'center'};
 widths.forEach((v,i)=>sheet.getRange(`${col(i)}${row}:${col(i)}${row+rows.length}`).format.columnWidth=v);
 return row+rows.length;}
function formula(sheet,cell,value){sheet.getRange(cell).formulas=[[value]];}
function num(sheet,range,format='0.000000'){sheet.getRange(range).setNumberFormat(format);sheet.getRange(range).format.horizontalAlignment='right';}
heading(d,'A2','原始阶段计时与计费边界');
const detailLast=table(d,5,['编号','任务','阶段含义','原stage名称','计入投影','原family','三轮合计 ms','阶段调用次数','ms/有效输入'],stages.map(g=>[g.key,names[g.key],descriptions[g.family]||g.family,g.stage,g.charged?'是':'否',g.family,g.ms,g.calls,null]),[10,24,35,55,14,24,20,18,19]);
for(let i=6;i<=detailLast;i++)formula(d,`I${i}`,`=G${i}/600`);
num(d,`G6:G${detailLast}`);num(d,`H6:H${detailLast}`,'#,##0');num(d,`I6:I${detailLast}`);d.getRange(`G6:G${detailLast}`).format.fill=colors.blue;d.getRange(`I6:I${detailLast}`).format.fill=colors.green;
d.getRange(`C6:D${detailLast}`).format={wrapText:true,rowHeight:46};d.freezePanes.freezeRows(5);d.freezePanes.freezeColumns(2);
heading(refsSheet,'A2','来源文件和SHA256');
table(refsSheet,5,['相对仓库路径','字节数','SHA256','用途'],refs.map(x=>[x.path,x.bytes,x.sha256,x.kind]),[100,16,70,24]);
refsSheet.getRange(`A6:A${5+refs.length}`).format={wrapText:true,rowHeight:52};

heading(s,'A2','Ours光学时间和电子时间分解');
s.getRange('A3').values=[['在线每有效输入。光学列由固定参数推导，电子列来自原三轮600输入。原前两份工作簿保持不变。']];
table(s,5,['编号','任务及原生单位','周期/有效输入','SLM ms','曝光 ms','额外CCD ms','光学小计 ms','电子计费 ms','投影总时间 ms','光学占比','成本状态'],records.map(r=>[r.id,r.task+(r.id.startsWith('01')?' 每视频':r.id==='04'?' 每标题':' 每样本'),null,null,null,null,null,null,null,null,r.calls?'三轮600输入已齐':'光学固定周期可推导，总时间待测']),[9,30,18,15,15,17,18,19,21,16,38]);
heading(s,'A18','一个光学周期的组成');
table(s,20,['环节','ms/周期','性质','当前参数覆盖范围'],[
 ['SLM',opt.slm_ms,'固定参数','SLM时序合计，未独立拆出通信、刷新、稳定各项'],
 ['CCD曝光',opt.exposure_ms,'固定参数','曝光积分窗口'],
 ['额外CCD',opt.additional_ccd_ms,'固定参数','原确认的额外CCD开销，未独立拆出读出和传输'],
 ['合计',null,'参数求和','不是新测的物理装置端到端时间或纯光飞行时间']],[30,18,18,70]);
formula(s,'B24','=SUM(B21:B23)');num(s,'B21:B24');s.getRange('B21:B23').format.fill=colors.amber;
heading(s,'A28','每轮200有效输入的光学执行计划');
table(s,30,['编号','任务','有效输入数','视频/整场','整场次数','周期/整场','周期/轮','周期/有效输入','等效帧光学ms','状态'],records.map(r=>[r.id,r.task,200,r.id==='01a'?16:1,null,['04','05','06'].includes(r.id)?3:6,null,null,null,r.calls?'已与原成本核对':'计算图计划，非成本完成']),[9,30,18,15,15,17,18,19,21,38]);
for(let i=0;i<records.length;i++){
 const r=records[i],row=6+i,p=31+i;
 formula(s,`E${p}`,`=ROUNDUP(C${p}/D${p},0)`);formula(s,`G${p}`,`=E${p}*F${p}`);formula(s,`H${p}`,`=G${p}/C${p}`);
 formula(s,`I${p}`,r.id.startsWith('01')?`=H${p}*$B$24/4`:'="不适用"');
 formula(s,`C${row}`,`=H${p}`);formula(s,`D${row}`,`=C${row}*$B$21`);formula(s,`E${row}`,`=C${row}*$B$22`);formula(s,`F${row}`,`=C${row}*$B$23`);formula(s,`G${row}`,`=SUM(D${row}:F${row})`);
 formula(s,`H${row}`,r.calls?`=K${67+records.filter(x=>x.calls).findIndex(x=>x.id===r.id)}`:'="待测"');
 formula(s,`I${row}`,`=IF(ISNUMBER(H${row}),SUM(G${row}:H${row}),"待测")`);formula(s,`J${row}`,`=IF(ISNUMBER(I${row}),G${row}/I${row},"待测")`);
}
num(s,'C6:I14');num(s,'J6:J14','0.00%');num(s,'C31:G39','#,##0');num(s,'H31:I39');s.getRange('D6:G14').format.fill=colors.amber;s.getRange('H6:H14').format.fill=colors.blue;s.getRange('I6:J14').format.fill=colors.green;s.getRange('K6:K14').format.wrapText=true;s.getRange('K6:K14').format.rowHeight=42;
heading(s,'A43','软件全窗诊断，不与上方物理光学小计重复相加');
const done=records.filter(r=>r.calls);
table(s,45,['编号','任务','软件全窗 ms','GPU FFT ms','并行残差 ms','layout和fusion ms','未打点软件开销 ms','其余计费电子 ms'],done.map(r=>[r.id,r.task,null,null,null,null,null,null]),[9,30,18,15,15,22,24,24]);
for(let i=0;i<done.length;i++){
 const row=46+i,task=`$A${row}`,sum=family=>`SUMIFS('电子与仿真明细'!$G$6:$G$${detailLast},'电子与仿真明细'!$A$6:$A$${detailLast},${task}${family?`,'电子与仿真明细'!$F$6:$F$${detailLast},"${family}"`:''})/600`;
 formula(s,`C${row}`,`=${sum()}`);formula(s,`D${row}`,`=${sum('simulation_only')}`);formula(s,`E${row}`,`=${sum('parallel_residual')}`);formula(s,`F${row}`,`=${sum('layout')}+${sum('fusion')}`);formula(s,`G${row}`,`=${sum('software_remainder')}`);
 formula(s,`H${row}`,`=C${row}-SUM(D${row}:G${row})`);
}
num(s,'C46:H51');s.getRange('C46:H51').format.fill=colors.grey;
const notes=[
 '计费关系：投影总时间＝固定光学周期时间＋已选电子阶段＋全部199真实请求间隙/轮。',
 'GPU FFT是光传播的数值仿真替身。上方用固定光学周期替换它，不把二者相加。',
 '电子CCD池化、归一化和可学习读出属于GPU/CPU计算，与0.0307 ms额外CCD硬件参数不同。',
 'Temporal 200视频为13个16视频整场，尾场8个padding槽也收费。每视频光学0.407433 ms，等效帧再除4。',
 'Spatial每视频包含4采样帧但按一个原整场收6周期。不要将每视频光学再乘4。',
 '文搜图在线仅语言3周期。离线图片库每图6周期不计入在线查询；图搜文离线标题库不计入在线图像查询。',
 '0.714/0.300/0.0307 ms来自既有人工确认参数，精确手册页未绑定。纯飞行、读出、传输无法进一步独立拆出，不填假0。',
 '蓝色为原计时汇总，橙色为固定参数推导，绿色为混合投影结果。模型加载、预制图库和最终文件写盘不计入在线。'];
notes.forEach((text,i)=>s.getRange('A'+(55+i)).values=[[text]]);
heading(s,'A64','电子计费由哪些部分组成');
table(s,66,['编号','任务','文件处理和H2D ms','Embedding和位置 ms','真实merger ms','adapter和幅度编码 ms','路由电子运算 ms','CCD数字处理 ms','任务头/检索/CPU输出 ms','真实请求间隙 ms','电子合计 ms'],done.map(r=>[r.id,r.task,null,null,null,null,null,null,null,null,null]),[9,30,18,15,15,22,24,24,21,38,38]);
const families=[['input'],['frontend'],['bridge'],['adapter','amplitude_encoding_nn'],['router'],['ccd_nonlinearity','ccd_readout'],['head','search','output'],['request_gap']];
for(let i=0;i<done.length;i++){
 const row=67+i;
 families.forEach((values,j)=>formula(s,`${col(j+2)}${row}`,'='+values.map(f=>`SUMIFS('电子与仿真明细'!$G$6:$G$${detailLast},'电子与仿真明细'!$A$6:$A$${detailLast},$A${row},'电子与仿真明细'!$F$6:$F$${detailLast},"${f}")/600`).join('+')));
 formula(s,`K${row}`,`=SUM(C${row}:J${row})`);
}
num(s,'C67:K72');s.getRange('C67:J72').format.fill=colors.blue;s.getRange('K67:K72').format.fill=colors.green;

heading(o,'A2','各任务的光路逐步骤时间');
o.getRange('A3').values=[['按组成列示，不表示执行先后。每步骤一次原传播/曝光周期；按有效输入摊销，不含电子编码、路由计算或CCD数字读出。']];
const opticalRows=[];
for(const r of records){const modes=r.id==='04'?['语言']:['05','06'].includes(r.id)?['视觉']:['视觉','语言'];for(const mode of modes)for(const role of ['路由','专家','全局'])opticalRows.push([r.id,r.task,mode,role,200,r.id==='01a'?16:1,null,null,null,null,null,r.calls?'原三轮传播次数已核对':'固定计算图计划，非成本完成']);}
const olast=table(o,5,['编号','任务','光路','步骤','有效输入/轮','输入/整场','该步骤周期/轮','SLM ms/输入','曝光 ms/输入','额外CCD ms/输入','步骤小计 ms/输入','证据状态'],opticalRows,[9,26,12,12,19,17,21,19,20,23,25,37]);
for(let row=6;row<=olast;row++){
 formula(o,`G${row}`,`=ROUNDUP(E${row}/F${row},0)`);formula(o,`H${row}`,`=G${row}/E${row}*'任务时间拆分'!$B$21`);formula(o,`I${row}`,`=G${row}/E${row}*'任务时间拆分'!$B$22`);formula(o,`J${row}`,`=G${row}/E${row}*'任务时间拆分'!$B$23`);formula(o,`K${row}`,`=SUM(H${row}:J${row})`);
}
num(o,`H6:K${olast}`);o.getRange(`H6:K${olast}`).format.fill=colors.amber;o.getRange(`L6:L${olast}`).format={wrapText:true,rowHeight:42};o.freezePanes.freezeRows(5);o.freezePanes.freezeColumns(2);
for(const [sheet,range] of [[s,'A6:A14'],[s,'A31:A39'],[s,'A46:A51'],[s,'A67:A72'],[o,`A6:A${olast}`],[d,`A6:A${detailLast}`]])sheet.getRange(range).setNumberFormat('00');
w.recalculate();
for(let i=0;i<records.length;i++){
 const r=records[i],row=6+i,v=s.getRange(`G${row}:J${row}`).values[0];const expected=(r.id==='01a'?13/200:1)*(['04','05','06'].includes(r.id)?3:6)*opt.per_pass_ms;
 close(v[0],expected,'Calculated optical subtotal');if(r.calls){close(v[2],r.ms,'Prior immutable total');close(v[1]+v[0],r.ms,'Optical/electronic sum');}
 const optical=opticalRows.map((x,j)=>[x,j+6]).filter(([x])=>x[0]===r.id).reduce((sum,[,n])=>sum+o.getRange(`K${n}`).values[0][0],0);close(optical,expected,'Branch optical plan');
}
// Recalculation proof: alter only a fixed driver, then restore it before export.
const original=s.getRange('G8').values[0][0];s.getRange('B21').values=[[opt.slm_ms+.1]];close(s.getRange('G8').values[0][0],original+.6,'Live fixed-driver dependency');s.getRange('B21').values=[[opt.slm_ms]];w.recalculate();close(s.getRange('G8').values[0][0],original,'Restored approved parameter');
await fs.mkdir(OUT,{recursive:true});
const error=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000});
await fs.writeFile(path.join(OUT,'formula_scan.ndjson'),error.ndjson);
for(const [sheetName,range,label] of [['任务时间拆分','A1:K14','overview'],['任务时间拆分','A18:J39','parameters_plan'],['任务时间拆分','A43:H62','software'],['任务时间拆分','A64:K72','electronic_summary'],['光路逐步骤','A2:L17','optical'],['电子与仿真明细','A4:I15','electronic'],['原始来源','A2:D11','sources']]){
 const image=await w.render({sheetName,range,scale:1.4,format:'png'});await fs.writeFile(path.join(OUT,label+'.png'),new Uint8Array(await image.arrayBuffer()));}
const file='A100_光学与电子时间逐任务分解_20261025.xlsx';await(await SpreadsheetFile.exportXlsx(w)).save(path.join(OUT,file));
const audit={status:'passed',scope:'File-only detailed timing decomposition; six measured Ours triples, three pending tasks',no_model_GPU_or_rental_work:true,original_intervals_nonoverlapping:true,qualified_runs:18,qualified_calls:3600,retained_internal_gaps:3582,optical_step_rows:opticalRows.length,stage_rows:stages.length,fixed_parameters:opt,source_files:refs,source_git:execFileSync('git',['rev-parse','HEAD'],{cwd:REPO,encoding:'utf8'}).trim(),records:records.map((r,i)=>({id:r.id,task:r.task,measured:!!r.calls,values:s.getRange(`C${6+i}:J${6+i}`).values[0]})),sources_unchanged:refs.every(x=>digest(path.join(REPO,x.path))===x.sha256),output:{file,bytes:fss.statSync(path.join(OUT,file)).size,sha256:digest(path.join(OUT,file))}};
assert(audit.sources_unchanged,'Source changed during authoring');
await fs.writeFile(path.join(OUT,'timing_breakdown_audit.json'),JSON.stringify(audit,null,2));
console.log(JSON.stringify({output:OUT,workbook:audit.output,records:audit.records}));
