import ast,json
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_interval_transfer_20261024.py'


def functions():
    tree=ast.parse(ENTRY.read_text());ns={}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('interval_for','validate')],type_ignores=[]),str(ENTRY),'exec'),ns)
    return ns


def test_exact_integer_intervals_not_rounded_percentages():
    fn=functions()['interval_for'];assert fn(346,1.2)==[342,345]and fn(354,1.5)==[348,353]
    assert fn(346,1.3,minimum_gap_pp=1.)==[341,341]
    for depth in(4,6):functions()['validate'](json.loads((HERE/('lab_caltech20_L'+str(depth)+'_interval_profile_20261024.json')).read_text()))
    c=json.loads((HERE/'lab_caltech20_L6_follow_interval_profile_20261024.json').read_text());functions()['validate'](c)
    assert c['reuse_verified_teacher']and c['parent_correct']==342 and c['gradient_balance_multipliers']=={'KL_TRAIN_interval':4}
    c=json.loads((HERE/'lab_caltech20_L4_interval341_profile_20261024.json').read_text());functions()['validate'](c)
    assert c['selection_interval']==[341,341]and c['expected_selected_update']==46 and c['reproduce_history_SHA256']


def test_first_real_checkpoint_and_peak_disclosed_without_prediction_manipulation():
    source=(HERE/'lab_caltech20_readout_transfer_20261024.py').read_text()
    assert 'interval_met or time.monotonic()' in source and 'peak_observed_correct=peak' in source
    assert 'first_actual_training_checkpoint_in_user_interval' in source
    s=ENTRY.read_text();assert 'official_TEST_accuracy=None' in s and 'higher_existing_results_preserved=True' in s
    assert "keys={'linear.weight','linear.bias'}" in s and 'loop.validate_state(m,native,original,keys)'in s
    assert "source_roles=np.array(['original_TRAIN']*1600)" in s and 'student_direct_TEST_gradient=False'in s
    assert 'm.save(protected'not in s and 'all_non_readout_bitwise_frozen=True'in s


def test_windows_artifact_paths_fit_and_camera_lineage_is_not_homogeneous():
    base='E:/code/guest/2026OpticsMoE/LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/'
    for folder in('fixed_simulation_terminal_20261023/L6_interval_20261024','layerwise_L4_on_20261023/e2_actual300_20261024/fullG2_interval_20261024'):
        assert len(base+folder+'/student/gradient_pool.npz')<260
    s=ENTRY.read_text();assert "camera_family='DVP'if depth==6 and stage not in('4','5')else'TUCam'"in s
    assert 'count<8' in s and 'maxerr<=3e-6'in s
