"""Author-side generation of the editable LightGen2 inference guide."""
from pathlib import Path
import hashlib
import json

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PUBLIC = Path(__file__).resolve().parents[4] / 'LightGen2'


def runtime_reference(reference):
    task = Path(__file__).resolve().parents[1]
    profile = json.loads((task / 'configs/selected.json').read_text('utf-8'))
    record = profile['runtime_benchmark']
    directory = task / record['directory']
    summary = json.loads((directory / 'summary.json').read_text('utf-8'))
    protocol = json.loads((directory / 'protocol.json').read_text('utf-8'))
    timing = summary['projects'][record['project']]['program_seconds']
    for name, expected in protocol['input_and_source_sha256'].items():
        with (PUBLIC / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Runtime measurement identity differs: ' + name)
    if (reference['total_runtime_seconds'] != timing['median']
            or reference['device_name'] != summary['hardware']['device']
            or reference['videos'] != summary['videos_per_process']):
        raise ValueError('Reference results differ from the selected runtime measurement')
    return timing, summary['measured_processes'], protocol['warmup_processes']


def paragraph(document, text, style=None):
    return document.add_paragraph(text, style)


def heading(document, text):
    document.add_heading(text, level=1)


def code(document, text):
    for line in text.splitlines():
        row = document.add_paragraph(style='Code')
        row.add_run(line or ' ')


def table(document, headers, rows, widths):
    result = document.add_table(rows=1, cols=len(headers))
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    result.autofit = False
    for column, width in zip(result.columns, widths):
        column.width = Inches(width)
    for cell, width, text in zip(result.rows[0].cells, widths, headers):
        cell.width = Inches(width)
        cell.text = text
    for values in rows:
        cells = result.add_row().cells
        for cell, width, text in zip(cells, widths, values):
            cell.width = Inches(width)
            cell.text = text
    for row_index, row in enumerate(result.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for edge in ('top', 'left', 'bottom', 'right'):
                line = OxmlElement('w:' + edge)
                line.set(qn('w:val'), 'single')
                line.set(qn('w:sz'), '4')
                line.set(qn('w:color'), 'D9D9D9')
                borders.append(line)
            properties.append(borders)
            margins = OxmlElement('w:tcMar')
            for edge in ('top', 'left', 'bottom', 'right'):
                value = OxmlElement('w:' + edge)
                value.set(qn('w:w'), '70')
                value.set(qn('w:type'), 'dxa')
                margins.append(value)
            properties.append(margins)
            if row_index == 0:
                fill = OxmlElement('w:shd')
                fill.set(qn('w:fill'), 'E8EDF2')
                properties.append(fill)
            for item in cell.paragraphs:
                item.paragraph_format.space_after = Pt(1)
                item.paragraph_format.space_before = Pt(1)
                item.paragraph_format.line_spacing = 1.0
                for run in item.runs:
                    run.font.size = Pt(9)
                    run.font.bold = row_index == 0
        cannot_split = OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(cannot_split)
    repeat = OxmlElement('w:tblHeader')
    result.rows[0]._tr.get_or_add_trPr().append(repeat)
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return result


def page(document, title):
    document.add_page_break()
    heading(document, title)


def build_guide():
    reference = json.loads((PUBLIC / 'results/02_metrics.json').read_text('utf-8'))
    timing, repeats, warmups = runtime_reference(reference)
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(.65)
    section.bottom_margin = Inches(.65)
    section.left_margin = Inches(.7)
    section.right_margin = Inches(.7)
    normal = document.styles['Normal']
    normal.font.name = 'Arial'
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (('Title', 21), ('Heading 1', 15), ('Heading 2', 11)):
        style = document.styles[name]
        style.font.name = 'Arial'
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(9)
    if 'Code' not in document.styles:
        from docx.enum.style import WD_STYLE_TYPE
        document.styles.add_style('Code', WD_STYLE_TYPE.PARAGRAPH)
    style = document.styles['Code']
    style.font.name = 'Consolas'
    style.font.size = Pt(8.5)
    style.paragraph_format.line_spacing = 1
    style.paragraph_format.space_after = Pt(2)
    style.paragraph_format.left_indent = Inches(.1)
    document.core_properties.title = 'LightGen2 temporal video quality inference'
    document.core_properties.subject = 'Installation, inference, architecture and optical array formats'
    document.core_properties.author = 'LightGen2 contributors'
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run('LightGen2 user guide  ·  ')
    run.font.size = Pt(8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)

    document.add_heading('LightGen2 temporal video quality inference', 0)
    paragraph(document, 'User guide for the complete normalized RGB evaluation')
    paragraph(document, 'LightGen2 predicts a temporal mean opinion score (MOS) for each video. '
              'The distribution contains normalized RGB frames from all 558 original TEST videos '
              'in one NPZ file. One invocation evaluates these videos in 35 optical fields and '
              'exports the numerical fields from one selected group during that evaluation.')
    heading(document, 'Environment and installation')
    table(document, ['Item', 'Configuration'], [
        ('Python', 'Python 3.11'),
        ('Inference', 'PyTorch 2.6 or later, NumPy and SciPy'),
        ('Figures', 'Matplotlib 3.7 or later and Arial'),
        ('Device', 'CUDA when available, otherwise CPU'),
    ], [1.3, 5.45])
    code(document, 'python -m pip install -r requirements.txt')
    heading(document, 'Run the full evaluation')
    code(document, 'python infer.py')
    paragraph(document, 'Run this command in the extracted LightGen2 directory. The default '
              'input is data/test_rgb.npz and the default output directory is results. '
              'The checkpoint is weights/LightGen2.pt. Explicit device selection is available '
              'through --device cuda or --device cpu.')
    paragraph(document, 'The reference run scores 558 distinct videos, using 34 full fields and '
              'one field with 14 actual videos and two padding entries. The final two slots '
              'repeat the first two inputs to fill the coherent field. Exactly 558 actual '
              'predictions and their human temporal MOS labels enter the Spearman rank '
              f"correlation coefficient (SRCC). The reference SRCC is {reference['SRCC']:.6f}.")
    paragraph(document, f"On a shared {reference['device_name']}, {repeats} complete runs from an "
              f"extracted package had a median of {timing['median']:.3f} seconds "
              f"and a range of {timing['minimum']:.3f} to {timing['maximum']:.3f} seconds. "
              f"{warmups} complete warm-up preceded fresh Python processes. Timing covers imports, "
              'model and NPZ loading, all 558 predictions, metrics and figure export, with CUDA '
              'synchronized before completion. Outer ZIP extraction precedes timing. The supplied '
              'results come from the median run. Each invocation saves its own total_runtime_seconds '
              'in 02_metrics.json. Runtime varies with shared load, hardware and storage.')
    paragraph(document, 'Field 1 is captured during this same evaluation. Its sixteen slots '
              'can be matched to results/01_predictions.csv using field and slot numbers. '
              'To display another field, select a number from 1 to 35.')
    code(document, 'python infer.py --visualize-field 12')
    paragraph(document, 'Each invocation processes the complete input archive. Figures use '
              'Arial at 7 pt, with 8 pt titles. Floating-point kernels can produce small '
              'platform differences. The evaluation split was used in checkpoint selection.')

    page(document, 'Inputs outputs and network dimensions')
    paragraph(document, 'The NPZ stores four normalized RGB frames for each video. Before '
              'packaging, frame indices were sampled at fractions 0.1, 0.3667, 0.6333 and 0.9 '
              'of each video’s frame-index range. Frames were center-cropped to 65% of the '
              'shorter side and resized to 448 × 448. Normalization is (RGB / 255 − 0.5) / 0.5.')
    paragraph(document, 'The model starts from these normalized frames and executes the saved '
              'Qwen patch projection, position addition and fixed pooling. Each video retains '
              'its original sample ID and one human temporal MOS label.')
    table(document, ['Component', 'Per video representation and operation'], [
        ('Normalized RGB', '4 × 3 × 448 × 448'),
        ('vision_frontend', 'Patch projection and positions give 4 × 784 × 1024, then fixed means give 4 × 49 × 1024'),
        ('vision_adapter', 'LayerNorm(1024) → Linear(1024, 192) → 4 × 49 × 192'),
        ('vision_router and vision_experts', 'One Top2 choice per frame, four bias-free Linear(192, 192) experts → 4 × 49 × 192'),
        ('vision_global', 'Four 27 × 27 amplitude patches → propagation → 4 × 49 × 192'),
        ('vision_merger', 'Mean/max over tokens → LayerNorm(384) → Linear(384, 192) → ReLU → frame positions'),
        ('language_adapter', '38 × 2048 prompt embeddings → LayerNorm(2048) → Linear(2048, 192)'),
        ('Language sequence', 'Four visual summaries and 38 prompt tokens with positions → 42 × 192'),
        ('language_router and language_experts', 'Route all 42 tokens and combine four linear experts → 42 × 192'),
        ('language_global', '56 × 56 amplitude patch → propagation → 42 × 96'),
        ('readout', 'Flatten(4032) → Linear(4032, 32) → ReLU → Linear(32, 5) → softmax → MOS'),
    ], [1.85, 4.9])
    paragraph(document, 'LightGen2.forward(normalized_rgb) accepts '
              '[batch, 16, 4, 3, 448, 448]. prediction has shape [batch, 16] and '
              'quality_probabilities has shape [batch, 16, 5]. The five grades are '
              'Bad, Poor, Fair, Good and Excellent. Their probability-weighted checkpoint '
              'grade scores produce MOS. Higher values indicate better temporal quality.')

    page(document, 'How the two branches process a video')
    document.add_heading('Frozen visual frontend', 2)
    paragraph(document, 'FrozenVisionFront allocates placeholder buffers in its constructor. '
              'load_lightgen2 fills them with saved Qwen patch-projection weights, bias and '
              'position tensors. The normalized RGB frames are duplicated across two temporal '
              'slices and passed to the saved Conv3d kernel with stride (2, 16, 16). '
              'Its 28 × 28 patch output contains 784 tokens of width 1024.')
    paragraph(document, 'Position addition is followed by two fixed spatial means. The first '
              'mean over 2 × 2 patch groups gives 14 × 14 tokens. The second gives 7 × 7 '
              'tokens. These task-specific pooling operations yield 49 × 1024 per frame.')
    paragraph(document, 'The patch kernel uses bfloat16 precision. The input archive stores '
              'CPU-normalized values rounded to this precision in a float16 container. '
              'Every stored value converts back to the prepared bfloat16 value exactly. '
              'The normalization and frozen-frontend precision conversion are completed '
              'during data preparation.')
    document.add_heading('Routing and experts', 2)
    paragraph(document, 'A router applies channel LayerNorm, flattening, Linear, ReLU and '
              'Linear to obtain four logits. Softmax produces four probabilities. The two '
              'largest are selected and divided by their sum to combine two expert outputs. '
              'Vision routes one 49 × 192 frame sequence. Language routes the complete '
              '42 × 192 sequence containing visual summaries and prompt tokens.')
    paragraph(document, 'Each branch owns its router and four independent bias-free expert '
              'matrices. Their parameters are shared across all videos and slots within '
              'the branch. Every token within a routed item shares its Top2 choice.')
    document.add_heading('Optical encoding and detection', 2)
    paragraph(document, 'Vision encodes a frame through Linear(192, 27), ReLU, token-axis '
              'Linear(49, 27) and ReLU. Language applies Linear(192, 56), ReLU and row padding '
              'from 42 to 56. Each amplitude patch is divided by its root-mean-square amplitude.')
    paragraph(document, 'Both branches use a 518 × 518 canvas and a 478 × 478 active area. '
              'Sixteen videos form a 4 × 4 arrangement. Each vision slot contains four '
              'frame patches in a 2 × 2 arrangement. Each language slot contains one patch. '
              'All sixteen learned 111 × 111 phase tiles are applied in their physical positions.')
    paragraph(document, 'Angular-spectrum propagation operates on the whole coherent canvas. '
              'Detected intensity is divided by each region’s mean, pooled to 49 × 96 '
              'for vision or 42 × 96 for language, then passed through LayerNorm and ReLU. '
              'Vision adds Linear(96, 192). The language projection is folded into the readout. '
              'Keep input ordering, field composition and cyclic padding fixed when comparing scores.')

    page(document, 'Project files and normalized RGB archive')
    table(document, ['Path', 'Purpose'], [
        ('model.py', 'Layer definitions, frozen patch projection, optical processing and checkpoint loader'),
        ('infer.py', 'Complete NPZ evaluation and SRCC calculation with one selected field capture'),
        ('visualize.py', 'Plotting and field export helpers called by infer.py'),
        ('weights/LightGen2.pt', 'Weights, configuration, frozen frontend, prompt and grade scores'),
        ('data/test_rgb.npz', 'Normalized RGB for all 558 videos, ordered IDs and human temporal MOS'),
        ('results/', '558 predictions, metric summary, evaluation plots and selected-field arrays'),
        ('requirements.txt', 'Combined inference and plotting dependencies'),
        ('README.md and README.docx', 'Quick commands and this editable guide'),
        ('LICENSE NOTICE licenses/', 'Apache-2.0 terms and Qwen attribution'),
    ], [1.9, 4.85])
    paragraph(document, 'The input NPZ is read one field at a time. Fields 1–34 hold sixteen '
              'actual videos each. Field 35 holds fourteen. Padding is constructed in memory '
              'during inference and is excluded from both the predictions table and metrics.')
    table(document, ['NPZ key', 'Content'], [
        ('format_version', 'Scalar format identifier 1'),
        ('sample_ids', '558 original dataset-relative video identities in evaluation order'),
        ('target_mos', '558 human temporal MOS labels in the same order'),
        ('rgb_0001 to rgb_0034', 'float16 arrays [16,4,3,448,448], with normalized RGB values in [−1,1]'),
        ('rgb_0035', 'float16 array [14,4,3,448,448] for the final actual videos'),
        ('normalization and storage_precision', 'Recorded pixel normalization and frozen-frontend precision'),
        ('frame_fractions crop_fraction source', 'Sampling positions, crop fraction and dataset attribution'),
    ], [2.2, 4.55])
    code(document, "import numpy as np\nwith np.load('data/test_rgb.npz', allow_pickle=False) as data:\n    ids = data['sample_ids']\n    labels = data['target_mos']\n    first_field = data['rgb_0001']")

    page(document, 'Read the results in numbered order')
    table(document, ['Result', 'Content and scope'], [
        ('01_predictions.csv', 'One record per actual video with sample ID, one-based field/slot, human MOS and predicted MOS'),
        ('02_metrics.json', 'SRCC, video/field counts, complete program runtime and device'),
        ('03_prediction_vs_mos.png', '558 points comparing predicted and human temporal MOS'),
        ('04_input_frames.png', 'Four sampled frames from each slot in the selected field'),
        ('05_optical_expert_share.npz and .png', 'Top2 events, probabilities and shares for all 558 actual videos'),
        ('06_vision_mask.npz and .png', 'Complete vision phase canvas, all sixteen tiles and actual complex modulation'),
        ('07_vision_input_field.npz and .png', 'Vision incident field and its amplitude, intensity and modulated field'),
        ('08_vision_output_field.npz and .png', 'Vision propagated field, detector intensity and transfer function'),
        ('09_language_mask.npz and .png', 'Complete language phase canvas, tiles and modulation'),
        ('10_language_input_field.npz and .png', 'Language incident field and modulated field'),
        ('11_language_output_field.npz and .png', 'Language propagated field, intensity and transfer function'),
    ], [2.85, 3.9])
    paragraph(document, 'The selected field is observed while computing its predictions. '
              'Each field is evaluated once. The fields used for figures and their sample '
              'identities are recorded in the optical NPZ files and matched to 01_predictions.csv. '
              'Expert shares use all 558 actual videos and exclude padding.')
    paragraph(document, 'To choose another input archive, output directory or visualized field, '
              'use the same entry point with the corresponding arguments.')
    code(document, 'python infer.py --input data/test_rgb.npz --output results\npython infer.py --visualize-field 35 --device cuda')

    page(document, 'Optical numerical arrays and display')
    paragraph(document, 'Load numerical archives with numpy.load(path, allow_pickle=False). '
              'Fields and modulation use complex64. Intensities and phase use float32. '
              'Coordinates are zero-based pixels, indexed as [y,x]. CSV fields and slots are one-based.')
    table(document, ['Archive', 'Numerical keys'], [
        ('06 and 09 mask files', 'phase [518,518], phase_tiles [16,111,111], modulation [518,518], slot_origins [16,2]'),
        ('07 and 10 input files', 'field, amplitude, intensity and modulated_field, each [518,518]'),
        ('08 and 11 output files', 'field, intensity and transfer, each [518,518]'),
        ('All optical files', 'video_names [16], valid [16] and field_number'),
        ('05 expert share', 'sample_ids [558], vision_selected/probabilities [558,4,4], language equivalents [558,4], and counts/shares [4] per branch'),
    ], [1.85, 4.9])
    paragraph(document, 'Phase is in radians on [−π,π]. modulation stores the actual complex '
              'factor used by the forward. The incident field precedes modulation. '
              'modulated_field is the input passed to propagation. Output intensity is the '
              'squared magnitude of the propagated field before detector normalization.')
    code(document, "import numpy as np\nmask = np.load('results/06_vision_mask.npz', allow_pickle=False)\nsource = np.load('results/07_vision_input_field.npz', allow_pickle=False)\ntarget = np.load('results/08_vision_output_field.npz', allow_pickle=False)\nmodulated = source['field'] * mask['modulation']\nnp.testing.assert_allclose(modulated, source['modulated_field'], rtol=1e-6)\nreplayed = np.fft.ifft2(np.fft.fft2(modulated) * target['transfer'])")
    paragraph(document, 'Phase plots use a linear −π to π color scale. Input amplitude is '
              'peak-normalized for display. Output detector intensity is displayed linearly '
              'using its 99.8th percentile as the upper limit, saturating the brightest 0.2% '
              'of pixels. Cyan outlines mark the 478 × 478 active area. The NPZ records '
              'display_amplitude or display_intensity, display_scale and display_percentile '
              'where applicable. Original fields and intensities retain full numerical values.')
    paragraph(document, 'The source and LightGen2 weights are distributed under Apache-2.0. '
              'Qwen tensors originate from Qwen3-VL-2B-Instruct, revision '
              '89644892e4d85e24eaac8bacfd4f463576704203. Attribution is supplied in NOTICE and '
              'licenses/QWEN_LICENSE.txt. RGB frames and temporal MOS labels originate from '
              'LGVQ at github.com/zczhang-sjtu/UGVQ and retain applicable dataset terms.')

    page(document, 'Complete phase and normalized optical fields')
    paragraph(document, 'These panels show field 1 captured during the complete 558-video '
              'reference evaluation. Paired NPZ files contain the exact arrays used for plotting.')
    for kind, vision_number, language_number in (('mask', 6, 9), ('input_field', 7, 10), ('output_field', 8, 11)):
        item = document.add_paragraph()
        item.alignment = WD_ALIGN_PARAGRAPH.CENTER
        item.paragraph_format.space_after = Pt(3)
        for branch, number in (('vision', vision_number), ('language', language_number)):
            image = PUBLIC / f'results/{number:02d}_{branch}_{kind}.png'
            picture = item.add_run().add_picture(str(image), width=Inches(3.15))
            picture._inline.docPr.set('descr', branch.capitalize() + ' ' + kind.replace('_', ' '))
    for root in (document._element, document.styles.element):
        for border in root.xpath('.//w:pBdr'):
            border.getparent().remove(border)
    document.save(PUBLIC / 'README.docx')
    print(PUBLIC / 'README.docx')


if __name__ == '__main__':
    build_guide()
