"""Create the short inference guide using the bundled document runtime."""
from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak


REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO/'LightGen2_MLP'


def build():
    metrics = json.loads((PROJECT/'results/02_metrics.json').read_text('utf-8'))
    pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('ArialBold', 'C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='ArialBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('Body', fontName='Arial', fontSize=10.5, leading=15, spaceAfter=9))
    styles.add(ParagraphStyle('Section', fontName='ArialBold', fontSize=12, leading=16,
                              spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle('MainTitle', fontName='ArialBold', fontSize=19, leading=24, spaceAfter=17))
    styles.add(ParagraphStyle('Command', fontName='Courier', fontSize=10, leading=16,
                              backColor=colors.HexColor('#EEEEEE'), borderPadding=7,
                              spaceBefore=2, spaceAfter=14))
    styles.add(ParagraphStyle('Caption', fontName='Arial', fontSize=9, leading=12,
                              alignment=TA_CENTER, spaceAfter=12))
    contents = []

    def body(text):
        contents.append(Paragraph(text, styles['Body']))

    def heading(text):
        contents.append(Paragraph(text, styles['Section']))

    contents.append(Paragraph('Codes for LightGen-II MLP version', styles['MainTitle']))
    heading('Introduction')
    body('This is the code to evaluate temporal video quality with the MLP version of LightGen-II. '
         'We use the video frontend, Top2 routing, shared experts and five-grade readout from the '
         'reference model, and implement both vision and language global modules as two-layer MLPs '
         'with ReLU activation. Each video receives one predicted quality score.')
    heading('Environment')
    body('Python 3.8 or later is supported. The code uses PyTorch, NumPy, SciPy and Matplotlib. '
         'Inference automatically uses an available CUDA device and also supports CPU execution.')
    heading('Installation')
    body('Extract the project and open a terminal in the LightGen2_MLP folder. Install the dependencies '
         'in your Python environment with the following command.')
    contents.append(Paragraph('python -m pip install -r requirements.txt', styles['Command']))
    heading('How to execute')
    contents.append(Paragraph('python infer.py', styles['Command']))
    body('This command processes all 558 LGVQ video inputs in batches of 16 and saves the results '
         'in the results folder. The final batch contains 14 videos. The code pads that batch for '
         'inference and computes the metrics from the 558 original video predictions.')
    body('Video frames pass through the frozen frontend, vision routing, experts and global MLP, '
         'and the visual merger. The resulting features join the prompt tokens and pass through '
         'the language routing, experts and global MLP. The readout converts five quality-grade '
         'probabilities into a predicted mean opinion score. Reference scores are used to compute '
         'the Spearman rank correlation coefficient (SRCC).')
    heading('Files')
    body('<b>data/input_for_review.npz</b> contains the video frames used for inference. '
         '<b>data/reference_mos.csv</b> contains the corresponding human quality scores for evaluation.')
    body('<b>weights/LightGen2.pt</b> contains the model weights. '
         '<b>model.py</b> defines the network, <b>data.py</b> reads the inputs, '
         '<b>infer.py</b> runs inference, and <b>plots.py</b> generates the result figures.')
    body('<b>requirements.txt</b> lists the Python dependencies. <b>README.pdf</b> is this guide. '
         '<b>LICENSE</b> and <b>licenses/</b> contain the project and upstream license terms.')
    contents.append(PageBreak())
    heading('Results')
    body('<b>01_predictions.csv</b> lists every video ID, reference quality score and predicted score. '
         '<b>02_metrics.json</b> records the number of evaluated videos and SRCC. '
         '<b>03_prediction_vs_mos.png</b> compares predictions with human scores. '
         '<b>04_input_frames.png</b> shows the four input frames for the first batch of videos.')
    source = PROJECT/'results/03_prediction_vs_mos.png'
    image = ImageReader(str(source))
    width, height = image.getSize()
    display_width = 315
    contents.append(Image(str(source), width=display_width, height=display_width*height/width))
    contents.append(Paragraph('Predicted and human temporal quality scores for 558 LGVQ videos.', styles['Caption']))
    body(f'In summary, this MLP version achieves an SRCC of {metrics["SRCC"]:.2f} on the supplied '
         f'{metrics["videos"]} videos. The package provides one inference entry point and the '
         'corresponding per-video results for reproducibility.')
    heading('License')
    body('The project code is distributed under GPL-3.0-only. The included upstream Qwen components '
         'retain their Apache-2.0 terms and attribution in licenses/.')

    def page_number(canvas, document):
        canvas.setFont('Arial', 9)
        canvas.drawCentredString(document.pagesize[0]/2, 27, str(document.page))

    document = SimpleDocTemplate(str(PROJECT/'README.pdf'), pagesize=(595.28,841.89),
                                 rightMargin=51, leftMargin=51, topMargin=43, bottomMargin=46,
                                 title='Codes for LightGen-II MLP version', author='')
    document.build(contents, onFirstPage=page_number, onLaterPages=page_number)
    print(str(PROJECT/'README.pdf'))


if __name__ == '__main__':
    build()
