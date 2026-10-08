import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const finalPath = "C:/Users/Xml12/OneDrive/2026OpticsMoE/LightGenV2/reports/20260915_lgvq_topsw/ppt_output/LGVQ_power_energy_1MAC_2OP_one_slide_v2.pptx";
const outputPath = "C:/Users/Xml12/OneDrive/2026OpticsMoE/LightGenV2/reports/20260915_lgvq_topsw/ppt_build/final-slide-1.png";
const presentation = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const slide = presentation.slides.getItem(0);
const rendered = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(outputPath, new Uint8Array(await rendered.arrayBuffer()));
console.log(outputPath);
