// Four newly measured triples. Artifact-tool authors a new workbook, never old.
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const BASE=path.join(HERE,'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b');
const OUT=path.join(BASE,'LG_four_front_scopes_20261101_01');
const BOOK=path.join(OUT,'LGVQ_lab_four_frontend_scopes_20261101.xlsx');
const input=path.join(OUT,'independent_CPU_audit01.json');
const baseline=path.join(BASE,'LG_dynamic_FIFO16_20261101_01/workbook_sources01.json');
const sha=p=>crypto.createHash('sha256').update(fss.readFileSync(p)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error(m);};
const close=(a,b)=>ok(typeof a==='number'&&Math.abs(a-b)<Math.max(1e-7,Math.abs(b)*1e-9),`${a} != ${b}`);
ok(!fss.existsSync(BOOK),'Never overwrite delivered or user-modified workbook');
ok(sha(baseline)==='ad43903d48df722e8de51ce6ba87d014e299533c7477d78838953040423803a6','Pinned baseline source');
const d=JSON.parse(await fs.readFile(input,'utf8'));
const old=JSON.parse(await fs.readFile(baseline,'utf8'));
ok(d.status==='independent_CPU_four_scope_triples_passed'&&d.formal_scores_bitwise_equal===6144&&d.warm_scores_bitwise_equal===576,'Complete 12-round audit required');
ok(d.rounds.length===12&&old.trials.length===36&&old.baselines.length===12,'No missing rounds or baselines');
const wb=Workbook.create();
const main=wb.worksheets.add('主表'),detail=wb.worksheets.add('阶段均值'),raw=wb.worksheets.add('实测与等待'),trials=wb.worksheets.add('Baseline原值'),timeline=wb.worksheets.add('384场估算时间轴');
const sheets=[main,detail,raw,trials,timeline];
for(const s of sheets){s.showGridLines=false;s.getRange('A1:R65').format={font:{name:'Arial',size:10},rowHeight:25,verticalAlignment:'center'};}
const put=(s,r,values)=>s.getRangeByIndexes(r-1,0,values.length,values[0].length).values=values;
const f=(s,c,formula)=>s.getRange(c).formulas=[[formula]];
const col=i=>{let s='';for(i++;i>0;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const header=(s,r,n)=>s.getRangeByIndexes(r-1,0,1,n).format={fill:'#263C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:48,wrapText:true,horizontalAlignment:'center'};
const title=(s,cell,text)=>{s.getRange(cell).values=[[text]];s.getRange(cell).format.font={name:'Arial',size:15,bold:true};};
const labels={qwen:'Qwen3-VL-2B',glm:'GLM-4.6V-Flash',clip:'CLIP ViT-B/32',yolo:'YOLO11s',kimi:'Kimi-VL-A3B',deepseek:'DeepSeek-VL2 Full'};
const scopes=['V0 原三个CPU步骤','V1 加读取与解码BGR','V2 再加BGR转RGB','V3 再加中心65%裁剪'];

title(detail,'A2','LG阶段均值与平均收尾计算');
put(detail,3,[['CPU值是线程活跃工作量总和÷16÷32，对应理想均分。后端是每场均值；两者均不是完整经过时间。']]);
const phases=[['decode','文件读取/定位/解码BGR','CPU'],['RGB','BGR转RGB','CPU'],['crop','中心65%裁剪','CPU'],['resize','Resize/PIL','CPU'],['normalization','归一化','CPU'],['patch','Patch与BF16 cast','CPU'],['tokenizer','文字tokenization','CPU'],['emb_text','文字embedding','GPU'],['emb_conv3d','视觉Conv3D embedding','GPU'],['emb_position','位置embedding','GPU'],['adapter_visual_pool','视觉pool','GPU'],['adapter_visual','视觉adapter','GPU']];
for(let i=1;i<=6;i++){phases.push([`optics${i}`,`光学层${i}`,'固定光周期'],[`oeo${i}`,`OEO${i}`,'GPU']);if(i===3)phases.push(['merger','Merger','GPU'],['adapter_language','语言adapter','GPU']);}
phases.push(['decoder','Decoder与评分','GPU']);
ok(phases.length===27,'Stage layout');
put(detail,5,[['阶段','设备',...Array.from({length:4},(_,v)=>[1,2,3,'均值'].map(i=>`V${v} r${i}/ms`)).flat()]]);header(detail,5,18);
for(const [i,[key,label,device]] of phases.entries()){
  const r=6+i;put(detail,r,[[label,device]]);
  for(let v=0;v<4;v++){
    const c=2+v*4;
    for(let rep=1;rep<=3;rep++){
      const row=d.rounds.find(x=>x.variant===v&&x.repeat===rep);
      const value=i<6?(v===0&&i<3?'未在线测':row.CPU_stage_mean_ms[key]):key==='tokenizer'?row.tokenizer_mean_ms:row.backend_stage_mean_ms[key];
      detail.getCell(r-1,c+rep-1).values=[[value]];
    }
    f(detail,col(c+3)+r,`=AVERAGE(${col(c)}${r}:${col(c+2)}${r})`);
    if(v===0&&i<3)detail.getCell(r-1,c+3).values=[['未在线测']];
  }
}
put(detail,35,[['计入CPU均值P','ms/场']]);
put(detail,36,[['平均图像就绪后端Q','ms/轮次收尾']]);
put(detail,37,[['收尾摊销Q÷32','ms/场']]);
put(detail,38,[['均值流水P＋Q÷32','ms/16视频']]);
put(detail,40,[['文字分支工作量非零，但在图像CPU准备期间完成，不重复加入图像就绪后的Q。']]);
put(detail,41,[['CPU阶段只按版本所选范围相加。V1/V2实际仍执行未计入的RGB/裁剪，以保持原输入。']]);
put(detail,42,[['Q从视觉Conv3D起，包含选定电子段、六固定光周期、decoder。FFT、残差和纯fusion/layout不计。']]);
put(detail,43,[['P＋Q÷32仅在本轮均值排程证明CPU主导、后端可被覆盖时使用；12轮均需独立核验通过。']]);
for(let v=0;v<4;v++){
  const c=2+v*4;
  for(let j=0;j<3;j++){
    const letter=col(c+j),selected=[9,10,11,...Array.from({length:v},(_,i)=>6+i)];
    f(detail,letter+'35',`=SUM(${selected.map(r=>letter+r).join(',')})`);
    f(detail,letter+'36',`=SUM(${letter}14:${letter}32)`);
    f(detail,letter+'37',`=${letter}36/32`);f(detail,letter+'38',`=SUM(${letter}35,${letter}37)`);
  }
  for(const r of [35,36,37,38])f(detail,col(c+3)+r,`=AVERAGE(${col(c)}${r}:${col(c+2)}${r})`);
}
detail.getRange('A1:A43').format.columnWidth=31;detail.getRange('B1:B43').format.columnWidth=20;detail.getRange('C1:R43').format.columnWidth=18;detail.getRange('C6:R38').setNumberFormat('0.000000');
detail.getRange('A35:R38').format.fill='#EDF3F9';detail.freezePanes.freezeRows(5);detail.freezePanes.freezeColumns(2);

title(raw,'A2','真实软件观测与分段等待差额');
put(raw,3,[['真实CPU队列与FFT软件经过时间保留OS间隙。固定光估算不计这些等待；差额不能全部归因于磁盘。']]);
put(raw,5,[['版本','轮次','CPU队列wall÷32/ms','FFT软件wall÷32/ms','实测板卡J/16视频','实测板卡均W','估算控制器计费s/场','估算电子计费s/场','估算光器件计费s/场','固定组件估算J/场','Report SHA256']]);header(raw,5,11);
for(const [i,r] of d.rounds.entries()){
  const row=6+i,e=r.component_energy_estimate;
  put(raw,row,[[r.variant,r.repeat,r.actual_CPU_queue_ms16,r.actual_FFT_software_ms16,r.actual_FFT_board_J16,r.actual_FFT_board_mean_W,e.controller_s16,e.electronic_s16,e.optical_s16,null,r.report_sha256]]);
  f(raw,'J'+row,`=G${row}*$B$21+H${row}*$B$22+I${row}*$B$23`);
}
put(raw,20,[['组件功率设定','W']]);header(raw,20,2);
put(raw,21,[['LG控制器/小主机',41.388],['LG电子板卡',62.842],['光器件',80.358],['Baseline固定主机',338.2],['LG参考tokens/16视频',old.common_tokens16]]);
put(raw,27,[['版本','轮次','阶段','活跃CPU均值/ms每视频','worker经过均值/ms每视频','off-CPU/时钟差额/ms每视频','差额占经过时间','说明']]);header(raw,27,8);
let rr=28;
for(const r of d.rounds.filter(r=>r.variant!==0))for(const p of ['decode','RGB','crop']){
  const x=r.frontend_measurements[p];
  put(raw,rr,[[r.variant,r.repeat,p,x.mean_active_ms_per_video,x.mean_worker_elapsed_ms_per_video,null,null,'按512视频汇总；非16线程并行wall，含计时开销']]);
  f(raw,'F'+rr,`=E${rr}-D${rr}`);f(raw,'G'+rr,`=F${rr}/E${rr}`);rr++;
}
put(raw,57,[['原始收据SHA',d.raw_receipt_sha256],['CPU核验SHA',sha(input)],['Baseline源SHA',sha(baseline)],['实测范围说明','FFT软件运行不等于固定光路部署；LG J是固定组件估算，不是插座/CPU实测。']]);
raw.getRange('A1:B60').format.columnWidth=18;raw.getRange('C1:J60').format.columnWidth=26;raw.getRange('K1:K60').format.columnWidth=73;raw.getRange('H28:H54').format.columnWidth=57;raw.getRange('C6:J17').setNumberFormat('0.000000');raw.getRange('D28:F54').setNumberFormat('0.000000');raw.getRange('G28:G54').setNumberFormat('0.00%');raw.freezePanes.freezeRows(5);

put(trials,1,[['配置','模型','MPS/%','GPU','轮次','视频数','16视频组数','E2E总秒','自身tokens','GPU实测J','GPU＋固定主机J','Report SHA256']]);header(trials,1,12);
trials.getRange('A2:L37').values=old.trials.map(t=>[t.key,labels[t.model],t.cap,t.hardware,t.repeat,t.videos,t.groups,t.seconds,t.tokens,t.gpu_j,t.system_j,t.report_sha256]);
trials.getRange('A1:D37').format.columnWidth=26;trials.getRange('E1:K37').format.columnWidth=22;trials.getRange('L1:L37').format.columnWidth=73;trials.getRange('H2:K37').setNumberFormat('#,##0.000000');trials.getRange('I2:I37').setNumberFormat('#,##0');trials.freezePanes.freezeRows(1);

title(main,'A2','LGVQ四范围均值流水对照');
put(main,3,[['LG为16视频并行估算；baseline为原batch=1端到端实测。倍率为异范围数值比，不称物理E2E实测加速。']]);
put(main,5,[['模型/范围','GPU/显存','MPS/%','视频并行数','时间/ms每16视频','tokens/s','总J/16视频','总J÷时间/W','耗时比/行÷V0','能耗比/行÷V0','速率比/V0÷行']]);header(main,5,11);
for(let v=0;v<4;v++){
  const r=6+v,meanCol=col(5+4*v);
  put(main,r,[[`LightGen ${scopes[v]}`,'实验室A100 40GB',100,16]]);
  f(main,'E'+r,`='阶段均值'!${meanCol}38`);f(main,'F'+r,`='实测与等待'!$B$25/E${r}*1000`);
  f(main,'G'+r,`=AVERAGE('实测与等待'!J${6+3*v}:J${8+3*v})`);
}
for(const [i,b] of old.baselines.entries()){
  const r=10+i,key=JSON.stringify(b.key),sum=k=>`SUMIFS('Baseline原值'!$${k}$2:$${k}$37,'Baseline原值'!$A$2:$A$37,${key})`;
  put(main,r,[[labels[b.model],['kimi','deepseek'].includes(b.model)?'租机A100 80GB':'实验室A100 40GB',b.cap,1]]);
  f(main,'E'+r,`=${sum('H')}/${sum('G')}*1000`);
  f(main,'F'+r,`=IF(${sum('I')}=0,"n.a.",${sum('I')}/${sum('H')})`);
  f(main,'G'+r,`=${sum('K')}/${sum('G')}`);
}
for(let r=6;r<=21;r++){
  f(main,'H'+r,`=G${r}/E${r}*1000`);f(main,'I'+r,`=E${r}/$E$6`);
  f(main,'J'+r,`=G${r}/$G$6`);f(main,'K'+r,`=IF(F${r}="n.a.","n.a.",$F$6/F${r})`);
}
put(main,23,[['LG使用Qwen参考token分子；大模型各用自身实际token分子。CLIP/YOLO无适用语言分子，不编造tokens/s。']]);
put(main,24,[['LG能耗为既有固定组件分时估算；baseline总J为GPU实测积分＋338.2W固定主机，不是整机实测。']]);
put(main,25,[['四配置各自独立三轮，版本间差额也包含运行波动，不强制结果单调；单项开销见阶段均值与等待记录。']]);
title(main,'A27','四版本计算拆分');
put(main,29,[['版本','CPU均值P/ms','平均收尾Q/ms','Q÷32/ms','P＋Q÷32/ms','三轮样本SD/ms']]);header(main,29,6);
for(let v=0;v<4;v++){
  const r=30+v,c=col(5+4*v),start=col(2+4*v),end=col(4+4*v);
  put(main,r,[[scopes[v]]]);f(main,'B'+r,`='阶段均值'!${c}35`);f(main,'C'+r,`='阶段均值'!${c}36`);f(main,'D'+r,`='阶段均值'!${c}37`);f(main,'E'+r,`='阶段均值'!${c}38`);f(main,'F'+r,`=STDEV('阶段均值'!${start}38:${end}38)`);
}
put(main,36,[['实验室硬件','A100 PCIe 40GB / 250W；Intel Xeon Gold 5218R @2.10GHz'],['租机baseline硬件','A100 SXM4 80GB / 400W；AMD EPYC 7742'],['每配置正式统计','3轮，每轮32场，每场16视频/64帧；每轮另有3场真实预热'],['固定光周期','SLM1886Hz，曝光300us，附加读出30.7us；非光路实测'],['仍不计入','quality像素准备、H2D、CPU调度/off-CPU等待、输出返回、FFT、残差及纯fusion/layout']]);
main.getRange('A1:A40').format.columnWidth=36;main.getRange('B1:B40').format.columnWidth=26;main.getRange('C1:D40').format.columnWidth=17;main.getRange('E1:K40').format.columnWidth=26;main.getRange('E6:K21').setNumberFormat('#,##0.000');main.getRange('B30:F33').setNumberFormat('0.000000');main.getRange('A6:K9').format.fill='#EDF3F9';

put(timeline,1,[['版本','轮次','场序号','估算CPU就绪/ms','估算评分完成/ms','估算累计均值/ms每16视频','实测CPU就绪/ms','实测FFT评分完成/ms']]);header(timeline,1,8);
timeline.getRange('A2:H385').values=d.estimated_groups.map(g=>[g.variant,g.repeat,g.field,g.estimated_CPU_ready_ms,g.estimated_score_done_ms,g.cumulative_average_ms,g.actual_CPU_ready_ms,g.actual_FFT_score_done_ms]);
timeline.getRange('A1:H385').format={font:{name:'Arial',size:10},columnWidth:25,rowHeight:25,verticalAlignment:'center'};header(timeline,1,8);timeline.getRange('D2:H385').setNumberFormat('0.000000');timeline.freezePanes.freezeRows(1);
put(timeline,387,[['实测列从本轮正式软件窗口起计，包含FFT及OS间隙；估算列从理想均值流水起计，两者不能混加。']]);

wb.recalculate();
for(const v of d.variants){close(main.getRange('E'+(6+v.variant)).values[0][0],v.mean_ms16);close(main.getRange('G'+(6+v.variant)).values[0][0],v.component_energy_J16);close(main.getRange('F'+(30+v.variant)).values[0][0],v.sample_SD_ms16);}
for(const [i,b] of old.baselines.entries()){close(main.getRange('E'+(10+i)).values[0][0],b.time16_s*1000);close(main.getRange('G'+(10+i)).values[0][0],b.energy16_j);}
const before=main.getRange('E6').values[0][0],cell=detail.getRange('C9'),saved=cell.values[0][0];
cell.values=[[saved+1]];wb.recalculate();close(main.getRange('E6').values[0][0],before+1/3);cell.values=[[saved]];wb.recalculate();close(main.getRange('E6').values[0][0],before);
const oldPower=raw.getRange('B21').values[0][0],oldJ=main.getRange('G6').values[0][0],counted=Array.from({length:3},(_,i)=>raw.getRange('G'+(6+i)).values[0][0]).reduce((a,b)=>a+b,0)/3;
raw.getRange('B21').values=[[oldPower+1]];wb.recalculate();close(main.getRange('G6').values[0][0],oldJ+counted);raw.getRange('B21').values=[[oldPower]];wb.recalculate();close(main.getRange('G6').values[0][0],oldJ);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:2000});ok(!errors.ndjson.includes('"kind":"match"'),'Formula errors');
const views=[['主表','A1:K25','main.png'],['主表','A27:H40','formula.png'],['阶段均值','A1:J43','stages_v01.png'],['阶段均值','K5:R38','stages_v23.png'],['实测与等待','A1:K25','measured.png'],['实测与等待','A27:H54','waiting.png'],['Baseline原值','A1:K13','baselines.png'],['384场估算时间轴','A1:H13','timeline.png']];
for(const [sheetName,range,name] of views){const png=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(OUT,name),new Uint8Array(await png.arrayBuffer()));}
await(await SpreadsheetFile.exportXlsx(wb)).save(BOOK);
await fs.writeFile(path.join(OUT,'builder_checks01.json'),JSON.stringify({status:'exported_pending_visual_and_saved_XML_QA',sha256:sha(BOOK),builder_source_sha256:sha(fileURLToPath(import.meta.url)),audit_sha256:sha(input),baseline_source_sha256:sha(baseline),formula_perturbations_restored:true,baseline_trials_unchanged:36,LG_scopes:4,rounds:12,extra_warm_fields_each:3,mean_stage_estimate_not_measured_E2E:true},null,2),{flag:'wx'});
console.log(JSON.stringify({file:BOOK,sha256:sha(BOOK),variants:d.variants}));
