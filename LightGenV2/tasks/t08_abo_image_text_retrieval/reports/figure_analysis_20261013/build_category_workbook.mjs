// Author with the bundled @oai/artifact-tool runtime, not repo dependencies.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'simple_category_comparison');
const d = JSON.parse(await fs.readFile(path.join(dir, 'category_data.json'), 'utf8'));
const w = Workbook.create();
const s = w.worksheets.add('类别准确率');
const p = w.worksheets.add('商品归类');
const r = w.worksheets.add('逐图原始结果');
for (const sh of [s, p, r]) {
  sh.showGridLines = false;
}
s.tabColor = '#16835F';
s.getRange('A2').values = [['ABO 图搜文：类别准确率对比']];
s.getRange('A2').format.font = { name: 'Arial', size: 16, bold: true };
s.getRange('A4:H4').values = [['类别','商品数','测试图数','Ours 命中','Qwen 命中','Ours 准确率','Qwen 准确率','差值（百分点）']];
p.getRange('A1:K1').values = [['展示类别','原 label','商品 ID','完整标题','训练图数','测试图数','Ours 命中','Qwen 命中','Ours 准确率','Qwen 准确率','差值（百分点）']];
p.getRange('A2:H101').values = d.products.map(v => [v.category,v.label,v.product_id,v.title,v.train_n,v.test_n,v.ours_correct,v.qwen_correct]);
p.getRange('I2:K101').formulas = d.products.map((_,i) => { const a = i+2; return [`=G${a}/F${a}`,`=H${a}/F${a}`,`=(I${a}-J${a})*100`]; });
p.getRange('A2:A101').dataValidation = {rule: {type:'list', values:d.categories.map(v=>v.category)}};
p.getRange('A2:A101').format.fill = '#FFF2CC';
p.getRange('M1').values = [['归类依据：按官方标题人工整理，仅用于汇总展示。']];
p.getRange('M2').values = [['可编辑 A 列归类；类别准确率表会重算。']];
p.getRange('M3').values = [['来源：titles.csv、train.csv、test.csv；命中数由逐图预测汇总。']];
r.getRange('A1:H1').values = [['sample_id','原 label','商品 ID','展示类别','Ours Top-1 命中','Qwen Top-1 命中','Ours 来源','Qwen 来源']];
r.getRange('A2:H2401').values = d.queries.map(v => [v.sample_id,v.label,v.product_id,v.category,v.ours_correct,v.qwen_correct,'ours_same_weight_replay_predictions.csv','qwen_dynamic2048_predictions.csv']);
r.getRange('J1').values = [['来源：原始预测按 sample_id 与 test.csv 逐项核对；命中规则为正确商品而非同类。']];
// Formula-linked summary: edited display categories update counts and rates.
s.getRange('A5:H16').formulas = d.categories.map((v,i) => { const a=i+5; return [
  null,`=COUNTIFS('商品归类'!$A$2:$A$101,A${a})`,
  `=SUMIFS('商品归类'!$F$2:$F$101,'商品归类'!$A$2:$A$101,A${a})`,
  `=SUMIFS('商品归类'!$G$2:$G$101,'商品归类'!$A$2:$A$101,A${a})`,
  `=SUMIFS('商品归类'!$H$2:$H$101,'商品归类'!$A$2:$A$101,A${a})`,
  `=IF(C${a}=0,"n.a.",D${a}/C${a})`,`=IF(C${a}=0,"n.a.",E${a}/C${a})`,
  `=IF(C${a}=0,"n.a.",(F${a}-G${a})*100)`]; });
s.getRange('A5:A16').values = d.categories.map(v=>[v.category]);
s.getRange('A17').values = [['总体']];
s.getRange('B17:H17').formulas = [['=SUM(B5:B16)','=SUM(C5:C16)','=SUM(D5:D16)','=SUM(E5:E16)','=D17/C17','=E17/C17','=(F17-G17)*100']];
s.getRange('A19').values = [['口径：每类命中数 ÷ 每类测试图数；仍要求匹配正确商品。']];
s.getRange('A20').values = [['Qwen：冻结 Embedding-2B，动态长宽输入，2048D。']];
s.getRange('A21').values = [['Ours：同权重逐图回放 1931/2400；历史汇总 1934/2400 不用于类别图。']];
s.getRange('A22').values = [['历史 TEST 参与选模；本表不是独立新商品泛化评估。']];
for (const [sh,range,header] of [[s,'A4:H17','A4:H4'],[p,'A1:K101','A1:K1'],[r,'A1:H2401','A1:H1']]) {
  sh.getRange(range).format.font = {name:'Arial',size:11};
  sh.getRange(range).format.rowHeight = 24;
  sh.getRange(range).format.verticalAlignment = 'center';
  sh.getRange(range).format.wrapText = false;
  sh.getRange(header).format = {fill:'#264653',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:28,horizontalAlignment:'center',verticalAlignment:'center'};
}
s.getRange('A4:A17').format.columnWidth = 20;
s.getRange('B4:E17').format.columnWidth = 14;
s.getRange('F4:H17').format.columnWidth = 21;
s.getRange('B5:H17').format.horizontalAlignment = 'right';
s.getRange('F5:G17').setNumberFormat('0.00%');
s.getRange('H5:H17').setNumberFormat('+0.00;-0.00;0.00');
s.getRange('A17:H17').format.fill = '#EAF2ED';
s.getRange('A17:H17').format.font.bold = true;
s.getRange('A19:A22').format.font = {name:'Arial',size:11,color:'#52616B'};
p.getRange('A1:A101').format.columnWidth=20;
p.getRange('B1:B101').format.columnWidth=10;
p.getRange('C1:C101').format.columnWidth=18;
p.getRange('D1:D101').format.columnWidth=95;
p.getRange('D2:D101').format.wrapText=true;
p.getRange('A2:K101').format.rowHeight=44;
p.getRange('E1:H101').format.columnWidth=14;
p.getRange('I1:K101').format.columnWidth=21;
p.getRange('I2:J101').setNumberFormat('0.00%');
p.getRange('K2:K101').setNumberFormat('+0.00;-0.00;0.00');
p.getRange('M1:M3').format.columnWidth=90;
p.freezePanes.freezeRows(1);
r.getRange('A1:A2401').format.columnWidth=24;
r.getRange('B1:B2401').format.columnWidth=10;
r.getRange('C1:D2401').format.columnWidth=20;
r.getRange('E1:F2401').format.columnWidth=22;
r.getRange('G1:H2401').format.columnWidth=47;
r.getRange('J1').format.columnWidth=100;
r.freezePanes.freezeRows(1);
w.recalculate();
const inspect=await w.inspect({kind:'table',range:'类别准确率!A4:H17',include:'values,formulas',tableMaxRows:14,tableMaxCols:8,maxChars:12000});
await fs.writeFile(path.join(dir,'workbook_inspect.ndjson'),inspect.ndjson);
const expected = d.categories.map(v=>[v.category,v.product_n,v.test_n,v.ours_correct,v.qwen_correct,v.ours_accuracy,v.qwen_accuracy,v.difference_pp]);
for (let i=0;i<12;i++) {
  const values=s.getRange(`A${i+5}:H${i+5}`).values[0];
  for(let j=0;j<8;j++) if(typeof expected[i][j]==='number' ? Math.abs(values[j]-expected[i][j])>1e-10 : values[j]!==expected[i][j]) throw new Error(`Summary mismatch ${i},${j}: ${values[j]}`);
}
// Verify category editing updates totals correctly, then restore original input.
const original=p.getRange('A2').values[0][0];
p.getRange('A2').values=[[d.categories[1].category]];
w.recalculate();
if(s.getRange('B13').values[0][0]!==16 || s.getRange('B6').values[0][0]!==10) throw new Error('Category editing did not recalculate');
p.getRange('A2').values=[[original]];
w.recalculate();
const totals=s.getRange('B17:E17').values[0];
if(JSON.stringify(totals)!==JSON.stringify([100,2400,1931,1769])) throw new Error('Total count mismatch');
const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Formula error scan'});
await fs.writeFile(path.join(dir,'formula_error_scan.ndjson'),errors.ndjson);
for (const [sheetName,range,name] of [['类别准确率','A1:H22','excel_summary_preview.png'],['商品归类','A1:K7','excel_mapping_preview.png'],['逐图原始结果','A1:F9','excel_raw_preview.png']]) {
  const blob=await w.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(dir,name),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(w)).save(path.join(dir,'ABO_category_comparison.xlsx'));
console.log(JSON.stringify({output:path.join(dir,'ABO_category_comparison.xlsx'),totals}));
