import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/Xml12/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const payload = JSON.parse(await fs.readFile(path.join(dir, "calculated_summary.json"), "utf8"));
const rows = payload.rows;
const wb = Workbook.create();
const navy = "#0B5A88", blue = "#DCECF7", gray = "#E7EAED", orange = "#FCE4D6", green = "#E2F0D9";

function styleHeader(range) {
  range.format.fill = navy;
  range.format.font = { bold: true, color: "#FFFFFF" };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
}

function styleBody(range) {
  range.format.borders = { preset: "all", style: "thin", color: "#C9D2D9" };
  range.format.verticalAlignment = "center";
}

const summary = wb.worksheets.add("Summary");
summary.showGridLines = false;
const headers = ["No.", "Task", "Metric", "Ours perf.", "Baseline perf.", "Units", "Ours optical ms", "Ours CUDA ms", "Ours total ms", "Baseline total ms", "Ours J", "Baseline J", "Speedup ×", "Energy-eff. gain ×", "Energy reduction", "Ours unit/J", "Baseline unit/J", "Ours effective TOP", "Baseline achieved TOP", "Ours TOPS/W", "Baseline TOPS/W"];
summary.getRange("A1:U1").values = [headers];
styleHeader(summary.getRange("A1:U1"));
const values = rows.map(r => [
  r.number, r.task, r.metric, r.ours_performance_full_test, r.baseline_performance_full_test,
  r.workload_units, r.optical_time_ms, r.ours_electronic_cuda_event_mean_ms,
  r.ours_total_time_ms, r.baseline_total_time_ms_matched_workload,
  r.ours_total_energy_j, r.baseline_total_energy_j_matched_workload,
  r.speedup_x, r.energy_efficiency_gain_x, r.energy_reduction_fraction,
  r.ours_units_per_j, r.baseline_units_per_j,
  r.ours_effective_top ?? null, r.baseline_achieved_top_matched_workload ?? null,
  r.ours_effective_tops_per_w ?? null, r.baseline_achieved_tops_per_w ?? null,
]);
summary.getRange(`A2:U${rows.length + 1}`).values = values;
styleBody(summary.getRange(`A2:U${rows.length + 1}`));
for (let i = 2; i <= rows.length + 1; i++) summary.getRange(`A${i}:U${i}`).format.fill = i % 2 ? gray : blue;
summary.getRange(`M2:U${rows.length + 1}`).format.fill = orange;
summary.getRange(`D2:E${rows.length + 1}`).format.numberFormat = "0.0000";
summary.getRange(`G2:N${rows.length + 1}`).format.numberFormat = "0.000";
summary.getRange(`O2:O${rows.length + 1}`).format.numberFormat = "0.00%";
summary.getRange(`P2:U${rows.length + 1}`).format.numberFormat = "0.0000";
summary.freezePanes.freezeRows(1);
summary.getRange("A:A").format.columnWidth = 8;
summary.getRange("B:B").format.columnWidth = 28;
summary.getRange("C:C").format.columnWidth = 14;
summary.getRange("D:U").format.columnWidth = 15;
summary.getRange("A1:U20").format.wrapText = true;

const inputs = wb.worksheets.add("Inputs & formulae");
inputs.showGridLines = false;
inputs.getRange("A1:D1").values = [["Input / equation", "Value", "Unit", "Status"]];
styleHeader(inputs.getRange("A1:D1"));
const c = payload.constants;
inputs.getRange("A2:D11").values = [
  ["Optical devices", c.optical_devices_w, "W", "measured fixed input"],
  ["Control chassis", c.control_host_w, "W", "measured fixed input"],
  ["Ours A100", c.ours_a100_w, "W", "measured fixed input"],
  ["Baseline host", c.baseline_host_w, "W", "measured fixed input"],
  ["Physical pass", c.physical_pass_ms, "ms", "0.714+0.300+0.0307"],
  ["Ours energy", "topt×(80.358+41.388)+telec×(41.388+62.842)", "J", "derived"],
  ["Baseline energy", "twall×(338.2+A100 active mean)", "J", "derived"],
  ["Timing samples", 200, "test items", "sample 1 retained"],
  ["Explicit warm-up", 0, "forwards", "none"],
  ["Operation convention", "1 MAC = 2 OP", "", "MAC-derived"],
];
styleBody(inputs.getRange("A2:D11"));
inputs.getRange("A6:D8").format.fill = orange;
inputs.getRange("A:A").format.columnWidth = 28;
inputs.getRange("B:B").format.columnWidth = 55;
inputs.getRange("C:C").format.columnWidth = 15;
inputs.getRange("D:D").format.columnWidth = 24;
inputs.getRange("A1:D11").format.wrapText = true;

const power = wb.worksheets.add("Baseline power summary");
power.showGridLines = false;
power.getRange("A1:G1").values = [["No.", "Task", "A100 active mean W", "Chassis W", "Combined W used for energy", "Telemetry samples", "Sampling interval ms"]];
styleHeader(power.getRange("A1:G1"));
power.getRange(`A2:G${rows.length + 1}`).values = rows.map(r => [
  r.number, r.task, r.baseline_a100_active_mean_power_w, c.baseline_host_w,
  r.baseline_host_plus_a100_power_w, r.baseline_a100_active_power_sample_count,
  r.baseline_a100_power_sampling_interval_ms,
]);
styleBody(power.getRange(`A2:G${rows.length + 1}`));
for (let i = 2; i <= rows.length + 1; i++) power.getRange(`A${i}:G${i}`).format.fill = i % 2 ? gray : blue;
const avgRow = rows.length + 3;
power.getRange(`A${avgRow}:G${avgRow}`).values = [[
  "Mean", "Unweighted mean across seven reported rows",
  c.baseline_a100_active_mean_power_w_unweighted_across_rows,
  c.baseline_host_w,
  c.baseline_host_plus_a100_mean_power_w_unweighted_across_rows,
  "—", "—",
]];
power.getRange(`A${avgRow}:G${avgRow}`).format.fill = green;
power.getRange(`A${avgRow}:G${avgRow}`).format.font = {bold: true};
styleBody(power.getRange(`A${avgRow}:G${avgRow}`));
power.getRange(`C2:E${avgRow}`).format.numberFormat = "0.000";
power.freezePanes.freezeRows(1);
power.getRange("A:A").format.columnWidth = 10;
power.getRange("B:B").format.columnWidth = 32;
power.getRange("C:G").format.columnWidth = 22;
power.getUsedRange().format.wrapText = true;

function parseCsv(text) {
  const out = []; let row = [], cell = "", quote = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quote = false;
      else cell += ch;
    } else if (ch === '"') quote = true;
    else if (ch === ',') { row.push(cell); cell = ""; }
    else if (ch === '\n') { row.push(cell.replace(/\r$/, "")); out.push(row); row = []; cell = ""; }
    else cell += ch;
  }
  if (cell.length || row.length) { row.push(cell); out.push(row); }
  return out;
}

async function addCombinedCsvSheet(name, candidates, sourceLabel) {
  const sheet = wb.worksheets.add(name);
  sheet.showGridLines = false;
  const combined = [];
  let header = null;
  for (const item of candidates) {
    try {
      const parsed = parseCsv(await fs.readFile(item.file, "utf8"));
      if (!parsed.length) continue;
      if (!header) { header = [sourceLabel, ...parsed[0]]; combined.push(header); }
      for (const row of parsed.slice(1)) combined.push([item.label, ...row]);
    } catch (_) {}
  }
  if (!combined.length) combined.push([sourceLabel, "No data"]);
  sheet.getRange("A1").write(combined);
  styleHeader(sheet.getRangeByIndexes(0, 0, 1, combined[0].length));
  if (combined.length > 1) styleBody(sheet.getRangeByIndexes(1, 0, combined.length - 1, combined[0].length));
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.getUsedRange().format.wrapText = false;
  sheet.getUsedRange().format.autofitColumns();
  sheet.getUsedRange().format.autofitRows();
  sheet.getRange("A:A").format.columnWidth = 26;
}

const baselineTiming = [];
const baselinePower = [];
for (const r of rows) {
  const base = path.join(dir, r.baseline_report.replace(/report\.json$/, ""));
  for (const name of ["timing_per_sample.csv", "per_video_predictions_and_timing.csv"]) baselineTiming.push({label:r.number, file:path.join(base,name)});
  for (const name of ["power_utilization_samples.csv", "power_samples.csv"]) baselinePower.push({label:r.number, file:path.join(base,name)});
}
await addCombinedCsvSheet("Baseline raw timing", baselineTiming, "Task");
await addCombinedCsvSheet("Baseline raw power", baselinePower, "Task");
await addCombinedCsvSheet("Ours component timing", [{label:"all", file:path.join(dir,"raw_remote","ours","narrow_200_no_warmup","all_per_call_timings.csv")}], "Scope");

const prov = wb.worksheets.add("Provenance");
prov.showGridLines = false;
prov.getRange("A1:E1").values = [["Task", "Ours raw report", "Baseline raw report", "Timing policy", "Notes"]];
styleHeader(prov.getRange("A1:E1"));
prov.getRange(`A2:E${rows.length+1}`).values = rows.map(r => [r.number, r.ours_report, r.baseline_report, c.timing_policy, r.number === "05" ? "Performance uses the laboratory-server bound full-test result." : ""]);
styleBody(prov.getRange(`A2:E${rows.length+1}`));
prov.getRange("A:A").format.columnWidth = 10;
prov.getRange("B:C").format.columnWidth = 62;
prov.getRange("D:E").format.columnWidth = 45;
prov.getUsedRange().format.wrapText = true;

await wb.recalculate();
const inspect = await wb.inspect({kind:"workbook,sheet,formula", maxChars:12000, tableMaxRows:12, tableMaxCols:22});
await fs.writeFile(path.join(dir,"workbook_inspect.txt"), inspect.ndjson ?? String(inspect), "utf8");
const preview = await wb.render({sheetName:"Summary", autoCrop:"all", scale:1, format:"png"});
await fs.writeFile(path.join(dir,"workbook_summary_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(path.join(dir,"A100_demo_energy_efficiency_audit.xlsx"));
