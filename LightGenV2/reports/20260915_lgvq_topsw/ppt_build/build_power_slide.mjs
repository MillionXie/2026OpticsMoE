import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/Xml12/OneDrive/2026OpticsMoE";
const reportDir = path.join(workspaceDir, "LightGenV2/reports/20260915_lgvq_topsw");
const buildDir = path.join(reportDir, "ppt_build");
const outputDir = path.join(reportDir, "ppt_output");
const skillDir = "C:/Users/Xml12/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const pythonExecutable = "C:/Users/Xml12/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const finalPath = path.join(outputDir, "LGVQ_power_energy_1MAC_2OP_one_slide_v2.pptx");

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const font = "Microsoft YaHei";
const colors = {
  ink: "#152536",
  muted: "#607083",
  blue: "#0878C9",
  orange: "#E98524",
  green: "#138A68",
  red: "#C83E3E",
  grid: "#D9E1E8",
  pale: "#F4F7FA",
  white: "#FFFFFF",
};

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});
const slide = presentation.slides.add();
slide.background.fill = colors.white;

function addText(text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: opts.fill ?? "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: font,
    fontSize: opts.fontSize ?? 20,
    bold: opts.bold ?? false,
    color: opts.color ?? colors.ink,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "middle",
    autoFit: "shrinkText",
  };
  return shape;
}

addText("LGVQ 时间质量评价：功率、能耗与能效", 64, 38, 1152, 56, {
  fontSize: 34,
  bold: true,
});
addText("16 个视频，每视频 4 帧。统一采用 1 MAC = 2 OP", 66, 94, 720, 34, {
  fontSize: 18,
  color: colors.muted,
});

addText("实测功率与能量", 66, 144, 470, 34, {
  fontSize: 24,
  bold: true,
  color: colors.blue,
});

addText("Ours", 66, 190, 110, 30, { fontSize: 20, bold: true, color: colors.blue });
addText("1.162 J ÷ 8.660 ms = 134.180 W", 175, 186, 390, 38, {
  fontSize: 21,
  bold: true,
});
addText("Qwen3VL@A100", 66, 235, 180, 30, { fontSize: 18, bold: true, color: colors.orange });
addText("103.784 J ÷ 1200.053 ms = 86.483 W", 245, 231, 355, 38, {
  fontSize: 19,
  bold: true,
});
addText("Ours 的平均功率更高，但推理时间缩短，因此总能耗显著降低。", 66, 278, 510, 48, {
  fontSize: 17,
  color: colors.muted,
});

addText("运算量与计算能效", 66, 342, 470, 34, {
  fontSize: 24,
  bold: true,
  color: colors.blue,
});
addText("Ours", 66, 388, 95, 28, { fontSize: 19, bold: true, color: colors.blue });
addText("0.627584 TOP ÷ 1.162 J = 0.5401 TOPS/W", 155, 382, 430, 42, {
  fontSize: 19,
  bold: true,
});
addText("A100 基线", 66, 431, 120, 28, { fontSize: 18, bold: true, color: colors.orange });
addText("39.202220 TOP ÷ 103.784 J = 0.3777 TOPS/W", 185, 425, 410, 42, {
  fontSize: 18,
  bold: true,
});
addText("计算能效比  0.5401 ÷ 0.3777 = 1.43×", 66, 472, 510, 38, {
  fontSize: 21,
  bold: true,
  color: colors.green,
});

addText("任务级能效（视频/J，越高越好）", 644, 144, 540, 36, {
  fontSize: 24,
  bold: true,
  color: colors.blue,
  alignment: "center",
});

const chart = slide.charts.add("bar", {
  position: { left: 654, top: 188, width: 520, height: 292 },
  categories: ["任务能效"],
  series: [
    { name: "Ours", values: [13.7694], fill: colors.blue },
    { name: "Qwen3VL@A100", values: [0.1542], fill: colors.orange },
  ],
  barOptions: { direction: "column", grouping: "clustered" },
  hasLegend: true,
  legend: { position: "bottom" },
  dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0000" },
  valueAxis: {
    minimumScale: 0,
    maximumScale: 15,
    majorUnit: 5,
    numberFormatCode: "0",
    hasMajorGridlines: true,
    majorGridlines: { line: { fill: colors.grid, width: 1 } },
  },
  categoryAxis: { showLabels: false },
});
applyPresentationChartFont(chart, { fontFamily: font });

addText("13.7694 ÷ 0.1542 = 89.31×", 690, 486, 450, 42, {
  fontSize: 25,
  bold: true,
  color: colors.green,
  alignment: "center",
});

addText("138.57×", 74, 554, 210, 48, { fontSize: 31, bold: true, color: colors.blue, alignment: "center" });
addText("16 视频吞吐提升", 74, 600, 210, 28, { fontSize: 17, color: colors.muted, alignment: "center" });
addText("98.9%", 326, 554, 210, 48, { fontSize: 31, bold: true, color: colors.red, alignment: "center" });
addText("总能耗降低", 326, 600, 210, 28, { fontSize: 17, color: colors.muted, alignment: "center" });
addText("89.31×", 578, 554, 210, 48, { fontSize: 31, bold: true, color: colors.green, alignment: "center" });
addText("任务能效提升", 578, 600, 210, 28, { fontSize: 17, color: colors.muted, alignment: "center" });
addText("1.43×", 830, 554, 210, 48, { fontSize: 31, bold: true, color: colors.green, alignment: "center" });
addText("计算能效提升", 830, 600, 210, 28, { fontSize: 17, color: colors.muted, alignment: "center" });
addText("0.072625 J", 1038, 554, 190, 48, { fontSize: 24, bold: true, color: colors.ink, alignment: "center" });
addText("平均每视频", 1038, 600, 190, 28, { fontSize: 17, color: colors.muted, alignment: "center" });

addText("注：8.660 ms 为 16 视频并行批次延迟，0.072625 J/视频为该批次均摊值。Ours 报告 effective TOPS/W，A100 报告本工作负载 achieved TOPS/W。", 66, 660, 1150, 34, {
  fontSize: 13,
  color: colors.muted,
  alignment: "center",
});

slide.speakerNotes.textFrame.setText(
  "数据来源：LightGenV2/reports/20260915_lgvq_topsw/operation_count_and_topsw.json；" +
  "README_1MAC_2OP_CN.md。统一口径：1 MAC = 2 OP。" +
  "Ours：8.660 ms、1.162 J、0.627584397504 TOP。" +
  "Qwen3VL@A100：1200.053 ms、103.784 J、39.202219950240 TOP。"
);

const preview = await presentation.export({ slide, format: "png", scale: 1 });
await fs.writeFile(path.join(buildDir, "draft-slide-1.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(buildDir, "draft-slide-1.layout.json"), await layout.text());

const stagingDir = path.join(reportDir, ".codex-finalizer-power-slide");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
};
const fontPolicy = { basis: "design", families: [font] };

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "LGVQ_power_energy_1MAC_2OP_one_slide_v2.validation.json"),
});

console.log(JSON.stringify({ finalPath, result }, null, 2));
