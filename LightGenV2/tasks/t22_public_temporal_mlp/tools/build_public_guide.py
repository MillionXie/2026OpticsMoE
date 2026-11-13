"""Author-side generation of the editable LightGen2 inference guide."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PUBLIC = Path(__file__).resolve().parents[4] / 'LightGen2'


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
    paragraph(document, 'User guide for the standalone inference distribution')
    paragraph(document, 'LightGen2 predicts temporal video quality as a mean opinion score (MOS). '
              'The supplied model combines a frozen Qwen-derived visual frontend, electronic Top2 '
              'routers and linear experts, vision and language optical propagation, and a two-layer '
              'electronic readout. Sixteen videos share each coherent optical field.')
    heading(document, 'Environment and installation')
    table(document, ['Item', 'Configuration'], [
        ('Python', 'Python 3.11'),
        ('Inference', 'PyTorch 2.6 or later, OpenCV 4.10 or later, NumPy and SciPy'),
        ('Figures', 'Matplotlib 3.7 or later and an installed Arial font'),
        ('Execution', 'CPU or a CUDA-capable GPU with a compatible PyTorch build'),
    ], [1.3, 5.45])
    paragraph(document, 'Open a terminal in the extracted LightGen2 directory and install the dependencies.')
    code(document, 'python -m pip install -r requirements.txt')
    heading(document, 'Run the sixteen video demo')
    paragraph(document, 'The demo loads the first sixteen TEST videos listed in test_videos.csv '
              'and processes them together in the same order. Each clip contributes four sampled RGB frames.')
    code(document, 'python demo.py --device cpu\npython demo.py --device cuda')
    paragraph(document, 'Choose either CPU or CUDA. The command writes predictions.json, complete '
              'vision and language masks, and input/output optical fields to outputs/demo. '
              'The bundled results directory contains reference outputs from the same demonstration.')
    paragraph(document, 'The bundled demo_videos directory contains these sixteen LGVQ clips with '
              'their original dataset-relative paths. Their sample identities and temporal MOS '
              'labels are recorded in test_videos.csv. The --video-root argument can also point '
              'the demo to a complete local LGVQ videos directory.')
    paragraph(document, 'The distribution contains all frozen frontend tensors and the fixed-prompt '
              'embedding inside weights/LightGen2.pt. After installing dependencies, inference '
              'uses the local files directly.')

    page(document, 'Inputs outputs and network dimensions')
    paragraph(document, 'infer.py decodes four frames at evenly spaced fractions from 0.1 to 0.9 of '
              'each video’s frame-index range. Each frame is converted to RGB, center-cropped to '
              '65% of the shorter side, and resized to 448 × 448 with area interpolation. '
              'The frontend receives uint8 pixels.')
    table(document, ['Component', 'Per video representation and operation'], [
        ('Pixels', '4 × 3 × 448 × 448'),
        ('vision_frontend', 'Patch projection and positions give 4 × 784 × 1024, and fixed means yield 4 × 49 × 1024'),
        ('vision_adapter', 'LayerNorm(1024) → Linear(1024, 192) → 4 × 49 × 192'),
        ('vision_router and vision_experts', 'One Top2 decision per frame with four bias-free Linear(192, 192) experts, giving 4 × 49 × 192'),
        ('vision_global', '4 × 49 × 192 → four 27 × 27 patches → optical propagation → 4 × 49 × 192'),
        ('vision_merger', 'Mean and max over 49 tokens → 4 × 384 → LayerNorm → Linear(384, 192) → ReLU → add frame positions'),
        ('language_adapter', '38 × 2048 prompt embeddings → LayerNorm(2048) → Linear(2048, 192)'),
        ('Language sequence', 'Concatenate 4 visual summaries and 38 prompt tokens and add positions to give 42 × 192'),
        ('language_router and language_experts', 'Route all 42 tokens with four bias-free Linear(192, 192) experts, giving 42 × 192'),
        ('language_global', '42 × 192 → 56 × 56 per video → optical propagation → 42 × 96'),
        ('readout', 'Flatten(4032) → Linear(4032, 32) → ReLU → Linear(32, 5) → softmax → MOS expectation'),
    ], [1.85, 4.9])
    paragraph(document, 'The pixel API accepts [batch, 16, 4, 3, 448, 448]. The saved-embedding API '
              'accepts [batch, 16, 4, 49, 1024]. prediction has shape [batch, 16], and '
              'quality_probabilities has shape [batch, 16, 5].')
    paragraph(document, 'The five output grades are Bad, Poor, Fair, Good and Excellent, in this '
              'order. MOS is the probability-weighted expectation of their checkpoint grade scores. '
              'Higher MOS indicates better predicted temporal quality.')

    page(document, 'How the two branches process a video')
    document.add_heading('Frozen visual frontend', 2)
    paragraph(document, 'FrozenVisionFront allocates placeholder buffers in its constructor. '
              'load_lightgen2 then fills them with the saved Qwen patch-projection weights, bias '
              'and interpolated position tensors. Pixel values are scaled to [−1, 1]. Each frame '
              'is duplicated across two temporal slices and passed through the saved Conv3d kernel '
              'with stride (2, 16, 16). Its 28 × 28 patch output contains 784 tokens of width 1024.')
    paragraph(document, 'After position addition, a fixed mean over each 2 × 2 patch group gives '
              '14 × 14 tokens. A second 2 × 2 spatial mean gives 7 × 7 tokens. These pooling '
              'operations are the task-specific token layout. The resulting 49 × 1024 tensor '
              'is the input to vision_adapter.')
    document.add_heading('Routing and experts', 2)
    paragraph(document, 'Within each branch, a router normalizes token channels, flattens the '
              'sequence, and applies Linear → ReLU → Linear to obtain four logits. Softmax '
              'produces four probabilities. The two largest probabilities are selected and '
              'divided by their sum. Their two expert outputs are combined with these weights.')
    paragraph(document, 'The vision router receives one 49 × 192 frame sequence. Its decision '
              'is shared by all 49 tokens in that frame. The language router receives the '
              'complete 42 × 192 sequence containing four visual summaries and 38 prompt tokens. '
              'Each branch owns its router and four independent expert matrices. Within a branch, '
              'the same matrices are reused across all videos and optical slots.')
    document.add_heading('Optical encoding and detection', 2)
    paragraph(document, 'The vision amplitude encoder applies Linear(192, 27) and ReLU, followed '
              'by a token-axis Linear(49, 27) and ReLU. Each frame becomes a 27 × 27 patch. '
              'The language encoder applies Linear(192, 56) and ReLU and pads 42 rows to 56. '
              'Each patch is divided by its root-mean-square amplitude before field packing.')
    paragraph(document, 'Each branch uses a 518 × 518 propagation canvas and a 478 × 478 active '
              'area. Sixteen video slots form a 4 × 4 arrangement. Vision places four frame '
              'patches in a 2 × 2 arrangement inside each slot. Language places one patch per slot. '
              'All sixteen learned 111 × 111 phase tiles are applied in their physical positions. '
              'The checkpoint defines the complex modulation and propagation settings.')
    paragraph(document, 'Angular-spectrum propagation operates on the entire coherent canvas. '
              'Detected intensity is divided by its region mean, pooled to 49 × 96 for vision '
              'or 42 × 96 for language, and passed through LayerNorm and ReLU. Vision adds '
              'a Linear(96, 192) projection. The language projection is folded into the readout.')
    paragraph(document, 'Optical coupling makes the output depend on field composition and slot '
              'order. Preserve the same video ordering, grouping and cyclic padding when '
              'comparing predictions. A field contains sixteen videos even when the final '
              'input group has fewer actual videos. Processing 558 videos uses 35 fields, '
              'with 14 actual videos and two padding entries in the final field. '
              'Only the 558 actual videos contribute to the evaluation.')

    page(document, 'Project files and reference outputs')
    table(document, ['Path', 'Purpose'], [
        ('model.py', 'Layer definitions, vision/language optical processing and load_lightgen2'),
        ('infer.py', 'Video decoding, 16-video grouping, JSON prediction and pixel preprocessing'),
        ('demo.py', 'Run the first sixteen TEST videos through the same inference graph'),
        ('visualize.py', 'Observe the actual forward and export optical NPZ/PNG files'),
        ('weights/LightGen2.pt', 'Weights, architecture configuration, frozen frontend, prompt and grade scores'),
        ('demo_videos/', 'Sixteen distinct input clips in optical-slot order'),
        ('test_videos.csv', 'Ordered 558-video evaluation identities, relative paths and temporal MOS labels'),
        ('results/', 'Reference arrays, figures and predictions for the bundled demonstration'),
        ('requirements.txt', 'Combined inference and visualization dependencies'),
        ('README.md and README.docx', 'Quick commands and this editable detailed guide'),
        ('LICENSE NOTICE licenses/', 'Apache-2.0 terms and Qwen attribution'),
    ], [1.9, 4.85])
    paragraph(document, 'The configuration used to construct the model is stored alongside its '
              'tensors in the checkpoint. infer.py uses weights/LightGen2.pt by default. '
              'The --weights argument selects another compatible checkpoint.')
    table(document, ['Output stems', 'Numerical and visual content'], [
        ('vision_mask and language_mask', 'NPZ stores the complete phase map, sixteen phase tiles and complex modulation. PNG shows all sixteen tiles.'),
        ('vision_input_field and language_input_field', 'NPZ stores the incident field, amplitude, intensity and modulated input. PNG shows incident amplitude.'),
        ('vision_output_field and language_output_field', 'NPZ stores the propagated field, intensity and transfer function. PNG shows output intensity.'),
        ('input_frames.png', 'All sixteen videos and the four sampled frames from each'),
        ('optical_expert_share.npz and .png', 'Selections, softmax probabilities, counts and selection shares for this sixteen-video demo'),
        ('predictions.json', 'One filename, field index, zero-based slot and MOS per video'),
    ], [2.5, 4.25])
    paragraph(document, 'Numerical optical files and their figures share the same stem. The masks '
              'use the complete 518 × 518 canvas. NPZ filenames and embedded video_names arrays '
              'preserve the input order. The reference predictions were generated with CUDA. '
              'Floating-point kernels and video decoders can produce small platform differences.')

    page(document, 'Numerical file format and custom inference')
    paragraph(document, 'Load NPZ files with numpy.load(path, allow_pickle=False). Fields and '
              'modulation use complex64. Intensities and phase use float32. Pixel coordinates '
              'are zero-based, with the array indexed as [y, x]. Phase is stored in radians '
              'on [−π, π].')
    table(document, ['Archive', 'Keys and shapes'], [
        ('*_mask.npz', 'phase [518,518], phase_tiles [16,111,111], modulation [518,518], slot_origins [16,2], video_names [16]'),
        ('*_input_field.npz', 'field [518,518], amplitude [518,518], intensity [518,518], modulated_field [518,518], video_names [16]'),
        ('*_output_field.npz', 'field [518,518], intensity [518,518], transfer [518,518], video_names [16]'),
        ('optical_expert_share.npz', 'vision_selected and vision_probabilities [16,4,4], language equivalents [16,4], and selection_count and share_percent [4] for each branch'),
    ], [1.85, 4.9])
    paragraph(document, 'phase contains the learned physical phase, with zero outside the tile '
              'supports. modulation is the actual complex factor used by the forward. '
              'The input archive stores the field before modulation and the modulated field '
              'passed into propagation. The output archive stores the propagated field before '
              'detector normalization. Its intensity is the squared field magnitude.')
    code(document, "import numpy as np\nmask = np.load('results/vision_mask.npz', allow_pickle=False)\nsource = np.load('results/vision_input_field.npz', allow_pickle=False)\ntarget = np.load('results/vision_output_field.npz', allow_pickle=False)\nmodulated = source['field'] * mask['modulation']\nnp.testing.assert_allclose(modulated, source['modulated_field'], rtol=1e-6)\nreplayed = np.fft.ifft2(np.fft.fft2(modulated) * target['transfer'])")
    paragraph(document, 'Phase figures use a linear −π to π color scale. Input figures show '
              'field amplitude with peak normalization. Output figures show native-pixel detector '
              'intensity scaled linearly by its 99.8th percentile. The brightest 0.2% of output '
              'pixels saturate in the displayed [0, 1] color scale. Cyan outlines mark the '
              '478 × 478 active area. '
              'Expert share is the fraction of '
              'Top2 selection events assigned to each expert in the displayed demo batch.')
    paragraph(document, 'Input archives also contain display_amplitude [518,518] and display_scale. '
              'Output archives contain display_intensity [518,518], display_scale and '
              'display_percentile. The original field and intensity arrays are stored at their '
              'full numerical values and native 518 × 518 resolution.')
    code(document, 'python infer.py --video-dir path/to/videos --device cuda\npython visualize.py --video-dir path/to/16_videos --device cuda')
    paragraph(document, 'Directory inputs are sorted by filename and include MP4, MOV, AVI and MKV. '
              'For an explicit order, use --videos followed by the paths with infer.py. '
              'visualize.py exports exactly one field from a directory containing sixteen videos. '
              'CPU is available through --device cpu. Figure export requires Arial.')
    paragraph(document, 'For the complete 558-video evaluation, retain the supplied test_videos.csv '
              'order and point --video-root to the LGVQ videos directory. The same infer.py '
              'entry processes 35 fields and writes the per-video MOS and overall SRCC to JSON.')
    code(document, 'python infer.py --evaluation-csv test_videos.csv '
         '--video-root path/to/LGVQ/videos --device cpu --output outputs/test.json')
    paragraph(document, 'The selected checkpoint yields SRCC 0.798898 on the 558-video cached-feature '
              'development evaluation. Direct video decoding on the validated CUDA runtime '
              'yields SRCC 0.798892. The evaluation split was used in checkpoint selection.')
    paragraph(document, 'LightGen2 is distributed under Apache-2.0. The embedded visual tensors '
              'and fixed-prompt embedding were extracted from Qwen3-VL-2B-Instruct, revision '
              '89644892e4d85e24eaac8bacfd4f463576704203. Upstream attribution and terms are '
              'provided in NOTICE and licenses/QWEN_LICENSE.txt.')
    paragraph(document, 'The evaluation identities and temporal MOS labels originate from LGVQ. '
              'Original videos are available through github.com/zczhang-sjtu/UGVQ.')

    page(document, 'Complete phase and normalized optical fields')
    paragraph(document, 'These panels show all sixteen video slots from the bundled demonstration. '
              'Their paired NPZ files contain the exact numerical arrays used for the figures.')
    for kind in ('mask', 'input_field', 'output_field'):
        item = document.add_paragraph()
        item.alignment = WD_ALIGN_PARAGRAPH.CENTER
        item.paragraph_format.space_after = Pt(3)
        for branch in ('vision', 'language'):
            picture = item.add_run().add_picture(str(PUBLIC / f'results/{branch}_{kind}.png'), width=Inches(3.15))
            picture._inline.docPr.set('descr', branch.capitalize() + ' ' + kind.replace('_', ' '))
    for root in (document._element, document.styles.element):
        for border in root.xpath('.//w:pBdr'):
            border.getparent().remove(border)
    document.save(PUBLIC / 'README.docx')
    print(PUBLIC / 'README.docx')


if __name__ == '__main__':
    build_guide()
