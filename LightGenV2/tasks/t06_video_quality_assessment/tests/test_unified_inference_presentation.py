"""CPU presentation contracts; no inference or file authoring."""
import ast
from pathlib import Path

TASK=Path(__file__).resolve().parents[1]


def test_new_plot_source_is_opt_in_and_old_resident_evidence_retained():
    text=(TASK/'reports/plot_selected_execution.py').read_text(encoding='utf-8')
    ast.parse(text)
    assert "'--unified-inference' in sys.argv" in text
    assert 'lgvq_unified_inference_recheck_20261018/presentation/' in text
    assert 'lgvq_resident_busy50_a100_20261017/attempt03_remaining_20261018/service_presentation/' in text
    assert 'No compliant dataset-once paced cost' in text
    assert 'mean of three separately paired energy estimates' in text


def test_workbook_uses_paired_repeat_energy_not_mean_power_product():
    text=(TASK/'reports/build_selected_execution.mjs').read_text(encoding='utf-8')
    assert "const book=unified?await SpreadsheetFile.importXlsx" in text
    assert "const resident=!unified&&x.duration_s!=null" in text
    assert "for(const col of ['G','H','K'])" in text
    assert 'AVERAGE(${refs.map' in text
    assert 'cost_wall_sum_ms' in text and 'excluded_wait_total_s' in text
    assert "if(driverFormula)inputs.getRange(driver).formulas=[[driverFormula]]" in text


def test_labels_keep_independent_utilization_only():
    tree=ast.parse((TASK/'reports/plot_selected_execution.py').read_text(encoding='utf-8'))
    label=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compact_point_label')
    source=ast.unparse(label)
    assert 'utilization_percent' in source and 'spatial' in source
    assert 'batch' not in source


def test_raw_audit_preserves_manifest_and_checks_exact_measured_source_blob():
    text=(TASK/'reports/plot_selected_execution.py').read_text(encoding='utf-8')
    assert 'for actual,metric_expected' in text
    assert 'raw_source.replace(b\'\\r\\n\',b\'\\n\')==blob' in text
    assert 'identity[\'git_commit\']+\':\'+p' in text
    assert "resumed['completed']==['glm_paced_b02_r01']" in text
    assert "last['completed']==['clip_paced_b10_r01','yolo_paced_b03_r01']" in text
