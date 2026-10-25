import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const out=process.argv[2];if(!out)throw Error('New immutable data folder required');
const d=JSON.parse(await fs.readFile(path.join(out,'independent_results.json'),'utf8'));
if(d.status!=='passed'||d.models.length!==6||d.serial_rows.length!==6144)throw Error('Complete audited evidence required');
const target=path.join(out,'A100_Temporal_Kimi_DS与LG优化_单轮2048次_20261026.xlsx');
try{await fs.access(target);if(!process.argv.includes('--rebuild-own'))throw Error('Preserve existing output')}catch(e){if(e.code!=='ENOENT')throw e;}
const w=Workbook.create();const s=w.worksheets.add('16视频对照'),c=w.worksheets.add('累计时间怎么算'),l=w.worksheets.add('LG逐组原值'),q=w.worksheets.add('大模型逐视频'),f=w.worksheets.add('原始文件索引');
const put=(sh,r,rows)=>sh.getRangeByIndexes(r-1,0,rows.length,rows[0].length).values=rows;
const title=(sh,r,text,last='G',height=40)=>{sh.getRange(`A${r}:${last}${r}`).merge();sh.getRange(`A${r}`).values=[[text]];sh.getRange(`A${r}:${last}${r}`).format={wrapText:true,rowHeight:height};};
const header=(sh,r,last='G')=>{sh.getRange(`A${r}:${last}${r}`).format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42,horizontalAlignment:'center'};};
for(const [sh,rows,cols] of [[s,41,7],[c,36,7],[l,390,12],[q,6150,13],[f,d.source_files.length+6,4]]){
  const area=sh.getRangeByIndexes(0,0,rows,cols);sh.showGridLines=false;area.format.font={name:'Arial',size:11,color:'#182B3A'};area.format.rowHeight=25;area.format.verticalAlignment='center';
  sh.getRange('A:A').format.columnWidth=31;sh.getRange('B:G').format.columnWidth=22;
}
s.tabColor='#243B53';c.tabColor='#7294AD';
title(s,1,'Temporal：新租机单轮与LG实现优化');s.getRange('A1').format.font={size:17,bold:true};
title(s,2,'每16视频＝64帧。2048次输入＝512个原TEST视频循环4次；每个模型仅一轮。', 'G',35);
title(s,3,'Qwen、Kimi、DS均为MPS50%逐视频串行；LG为MPS100%每场16视频相干并行。首批和计费电子建图成本均保留。','G',40);
put(s,5,[['模型 / 实现','实际平台','每16视频/ms','等效单帧/ms','除以LG低层捕获','除以LG纯评分','测量身份']]);header(s,5);
const labels=['保留的原Qwen50单轮','本次Kimi50新单轮','本次完整DS50新单轮','原33段图，完整辅助计算','减少重复Graph捕获准备','仅再省训练辅助统计'];
d.models.forEach((m,i)=>{
 const r=6+i;put(s,r,[[m.label,m.platform,null,null,null,null,labels[i]]]);
 s.getRange(`C${r}:F${r}`).formulas=[[`='累计时间怎么算'!G${6+i}`,`=C${r}/64`,`=C${r}/$C$10`,`=C${r}/$C$11`]];
});
s.getRange('C6:D11').setNumberFormat('0.000');s.getRange('E6:F11').setNumberFormat('0.00"×"');s.getRange('A6:G11').format={wrapText:true,rowHeight:46};s.getRange('A10:G11').format.fill='#E7F0F6';s.getRange('G:G').format.columnWidth=32;
title(s,13,'倍率解释：Qwen50与LG同为40GB PCIe卡；Kimi/DS来自80GB SXM4卡，其倍数是跨配置时间比，不能声称同硬件模型收益。','G',45);
title(s,14,'LG为“实际计费电子＋固定光周期”的混合投影，不是光路设备实测或视频文件端到端。双方均删除每条图像准备和原始CPU→GPU搬运。','G',45);
put(s,16,[['百倍目标检查','LG低层捕获','LG纯评分','单位','含义',null,null]]);header(s,16);
put(s,17,[['正式单轮倍率',null,null,'倍','保留全部2048次，不能改取最快窗口',null,null],['LG达到Qwen50百倍的上限',null,null,'ms/16视频','Qwen50每16视频 ÷ 100',null,null],['六次固定光周期',d.physical_fixed_time16_ms,d.physical_fixed_time16_ms,'ms/16视频','6×(1000/1400＋0.300＋0.0307)',null,null],['留给电子及其他等待的预算',null,null,'ms/16视频','目标上限减固定光周期',null,null],['本轮实际电子及其他等待',null,null,'ms/16视频','含首次计费电子建图',null,null],['完整后124场诊断倍率',null,null,'倍','排除启动段的辅助诊断，不当作正式结果',null,null]]);
s.getRange('B17:C17').formulas=[['=E6','=F6']];s.getRange('B18:C18').formulas=[['=$C$6/100','=$C$6/100']];
s.getRange('B20:C20').formulas=[['=B18-B19','=C18-C19']];s.getRange('B21:C21').formulas=[['=C10-B19','=C11-C19']];
s.getRange('B22:C22').formulas=[[`=$C$6/'累计时间怎么算'!B30`,`=$C$6/'累计时间怎么算'!B31`]];
s.getRange('B17:C22').setNumberFormat('0.000');s.getRange('B17:C17').setNumberFormat('0.00"×"');s.getRange('B22:C22').setNumberFormat('0.00"×"');s.getRange('E17:G22').merge(true);s.getRange('A17:G22').format={rowHeight:38,wrapText:true};
title(s,24,'已验证的优化：低层捕获只做一次同步、垃圾回收和显存缓存清理，后续仍逐段录制原操作。减少的是反复建图准备，不缓存特征或评分。','G',46);
title(s,25,'纯评分候选只略去训练loss、alignment及路由辅助统计。评分用的原网络、路由、六传播、融合和任务头均保留；辅助字典已缩减，不能称所有辅助输出也相同。','G',50);
title(s,26,'两个候选均逐位核对全部2048个评分。24.15与24.03 ms/16的差异不足1%，单轮不能证明纯评分候选有稳定额外收益。','G',42);
title(s,28,'后续优先项：减少空图提交和逐场强制同步，改有界异步排队；真正独立文字流可再验证。目前没有实测证明这些能补足到百倍。','G',44);
title(s,29,'异步后必须按实际依赖关键路径重新核算，不能从已经重叠的墙钟再次扣仿真，从而把CPU分词时间一起抹去。不得以换计费范围或筛慢Qwen造百倍。','G',46);
title(s,31,'未重跑原完整TEST。新轮评分与原参考逐位一致；512视频循环4次只是吞吐工作负载，不是2048个独立样本。新租机仅测时间，没有新增能耗结论。','G',45);

title(c,1,'累计时间怎么算：全量原窗口逐项加减');title(c,2,'先看第6—11行的最终累计与每16视频，再向下查原始扣项。时间全为毫秒。');
put(c,5,[['模型 / 实现','原观察窗口','图像准备/搬运扣项','计量记账扣项','三类GPU＋捕获扣项','最终累计时间','每16视频']]);header(c,5);
d.models.forEach((m,i)=>{const r=6+i;put(c,r,[[m.label,m.full_ms,m.image_H2D_ms,m.bookkeeping_ms,m.excluded_GPU_ms+m.excluded_capture_ms,null,null]]);c.getRange(`F${r}:G${r}`).formulas=[[`=B${r}-SUM(C${r}:E${r})+D${16+i}`,`=F${r}/128`]];});
c.getRange('B6:G11').setNumberFormat('0.000');c.getRange('A6:G11').format={rowHeight:42,wrapText:true};
put(c,15,[['明细（不再重复相加）','仿真/区域/残差GPU扣项','对应捕获CPU扣项','加回固定光周期','实际CPU文字分词','保留电子首次建图','计时原JSON绝对路径']]);header(c,15);
d.models.forEach((m,i)=>put(c,16+i,[[m.label,m.excluded_GPU_ms,m.excluded_capture_ms,m.fixed_ms,m.tokenizer_ms,m.electronic_capture_ms??null,m.source]]));
c.getRange('B16:F21').setNumberFormat('0.000');c.getRange('A16:G21').format={rowHeight:52,wrapText:true};c.getRange('G:G').format.columnWidth=64;
title(c,23,'Qwen/Kimi/DS：从第一条CPU准备开始至最后GPU评分就绪的连续跨度，减去每条“CPU图像准备＋原始H2D”，但实际文字tokenizer区间取并集并保留。','G',48);
title(c,24,'GPU图像卷积embedding、位置加入、合并、文字embedding、完整视觉/语言网络、完整词表投影及质量评分头，均保留。其他调度、同步、中间CPU结果返回等待也保留。','G',48);
title(c,25,'LG：原软件窗口减原始H2D、纯计量记账、六段仿真、六段探测区域处理、四段电子残差及三类专属捕获成本，再加入六次固定光周期。','G',48);
title(c,26,'区域处理指从模拟探测图裁区、归一化、拼接与能量统计。电残差是独立电子分支，不是光学直流。二者仍实际执行保输出，只按人工口径排除其计时。','G',48);
title(c,27,'区域后的学习型线性/非线性读出、路由、光电分支幅值归一化及融合、视觉语言merger和最终评分头继续计费；不是只算固定光周期。','G',46);
put(c,29,[['全后124场诊断','每16视频/ms','正式结果说明',null,null,null,null]]);header(c,29);
put(c,30,[[d.models[4].label,d.LG_diagnostics[1].post_capture_all124_time16_ms,'仅诊断全部后124场，不替代正式128场',null,null,null,null],[d.models[5].label,d.LG_diagnostics[2].post_capture_all124_time16_ms,'无选择最快子窗口',null,null,null,null]]);c.getRange('B30:B31').setNumberFormat('0.000');

title(l,1,'LG逐场原始账本：每场16视频、64帧','L');title(l,2,'第4场（编号3）含首次捕获。三方案各128场全部列出，不挑快场。','L');
put(l,5,[['实现','场号0起','原观察跨度/ms','原始H2D扣项/ms','记账扣项/ms','仿真GPU扣项/ms','区域处理扣项/ms','电残差扣项/ms','三类捕获扣项/ms','固定光周期/ms','本场投影/ms','等效帧/ms']]);header(l,5,'L');
put(l,6,d.LG_rows.map(r=>[...r.slice(0,10),null,null]));
l.getRange('K6:L389').formulas=d.LG_rows.map((r,i)=>[`=C${i+6}-SUM(D${i+6}:I${i+6})+J${i+6}`,`=K${i+6}/64`]);l.getRange('C6:L389').setNumberFormat('0.000');l.getRange('B:L').format.columnWidth=20;l.freezePanes.freezeRows(5);

title(q,1,'三大模型均逐视频串行：每条图像准备实际执行，核算逐条删除','M');title(q,2,'每模型相对其第一条CPU起点。CPU文字分词取调用区间并集，避免嵌套tokenizer重复计数。','M');
put(q,5,[['模型','序号0起','源视频0起','循环0起','原视频ID','CPU准备开始/ms','CPU准备结束/ms','H2D结束/ms','GPU模型开始/ms','GPU评分就绪/ms','CPU评分返回/ms','CPU文字分词/ms','图像准备/搬运扣项/ms']]);header(q,5,'M');
put(q,6,d.serial_rows.map(r=>[...r.slice(0,12),null]));q.getRange('M6:M6149').formulas=d.serial_rows.map((r,i)=>[`=H${i+6}-F${i+6}-L${i+6}`]);q.getRange('F6:M6149').setNumberFormat('0.000');q.getRange('B:D').format.columnWidth=15;q.getRange('E:E').format.columnWidth=54;q.getRange('F:M').format.columnWidth=21;q.freezePanes.freezeRows(5);

title(f,1,'原始路径、字节数与SHA256','D');title(f,2,'封存收据覆盖的文件重新核对远端SHA/size；本地审计、收据本体和外日志单列当前SHA，不冒称都由远端收据覆盖。','D',45);
put(f,5,[['本地原始文件完整路径','字节数','SHA256','核验身份']]);header(f,5,'D');put(f,6,d.source_files.map(x=>[x.path,x.bytes,x.sha256,x.verification]));f.getRange('A:A').format.columnWidth=105;f.getRange('B:B').format.columnWidth=17;f.getRange('C:C').format.columnWidth=72;f.getRange('D:D').format.columnWidth=37;f.getRange(`A6:D${5+d.source_files.length}`).format={wrapText:true,rowHeight:64};f.freezePanes.freezeRows(5);

w.recalculate();
const before=s.getRange('C6').values[0][0],input=c.getRange('B6').values[0][0];c.getRange('B6').values=[[input+128]];w.recalculate();if(Math.abs(s.getRange('C6').values[0][0]-before-1)>1e-6)throw Error('Recalculation failed');c.getRange('B6').values=[[input]];w.recalculate();
d.models.forEach((m,i)=>{if(Math.abs(s.getRange(`C${6+i}`).values[0][0]-m.time16_ms)>1e-5)throw Error('Result mismatch '+m.key);});
const inspect=await w.inspect({kind:'table',range:'16视频对照!A5:G11',include:'values,formulas',tableMaxRows:7,tableMaxCols:7,maxChars:5000});await fs.writeFile(path.join(out,'workbook_inspection.ndjson'),inspect.ndjson);
const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2500});console.log(errors.ndjson);
for(const [sh,range,name] of [[s,'A1:G14','summary'],[s,'A16:G22','target100'],[s,'A24:G31','improvements'],[c,'A5:G11','calculation'],[c,'A23:G31','scope'],[l,'A1:L12','LG'],[q,'A1:M11','serial'],[f,'A1:D9','sources']]){
 const b=await w.render({sheetName:sh.name,range,scale:1,format:'png'});await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await b.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(w)).save(target);console.log(target);
