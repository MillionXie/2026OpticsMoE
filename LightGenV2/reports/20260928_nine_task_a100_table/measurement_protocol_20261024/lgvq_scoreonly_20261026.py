"""Score-only inference on the validated low-level capture implementation.

Only training auxiliary alignment/loss/statistics are omitted. Prediction,
normalised prediction, router probabilities/top-two weights, all actual neural
branches, six optical propagations, fusion RMS and learned readouts retain their
original AST. Fusion's four diagnostics are still calculated. This is NOT a
claim that the complete original auxiliary output dictionaries are unchanged.
All replacement functions are scoped to this worker and restored afterwards.
"""

import ast
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType

import lgvq_lowlevel_capture_20261026 as lowlevel

projection = lowlevel.projection
ORIGINAL_INSTALL = projection.install_simulation_boundaries
OMITTED_OUTPUTS = (
    'optical_alignment_loss', 'router_balance_loss', 'router_importance_loss',
    'router_diversity_loss', 'router_capture_loss', 'guard_energy_loss',
)
RETAINED_OUTPUTS = ('prediction', 'normalized_prediction', 'routing',
                    'optical_enabled', 'quality_gate')


def _dump(node):
    return ast.dump(node, include_attributes=False)


def score_forward_AST(source):
    """A fail-closed allowlist, not a generic dead-code optimiser."""
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef) and function.name == 'forward'
    prediction_index = next(i for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'prediction')
    suffix = function.body[prediction_index+1:]
    assert len(suffix) == 2
    loss_if, returned = suffix
    assert isinstance(loss_if, ast.If) and isinstance(loss_if.test, ast.Name)
    assert loss_if.test.id == 'routing'
    assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Dict)
    keys = [key.value for key in returned.value.keys]
    assert set(keys) == set(RETAINED_OUTPUTS) | set(OMITTED_OUTPUTS)
    assigned = {target.id for node in ast.walk(loss_if) if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert assigned == {'balance', 'importance', 'diversity', 'capture', 'guard'}
    original_prefix = function.body[:prediction_index+1]
    removed = []
    counts = dict(alignments=0, guard_losses=0, initializers=0)
    class RemoveAuxiliary(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id in ('alignments', 'guard_losses'):
                assert isinstance(node.value, ast.List) and not node.value.elts
                counts['initializers'] += 1
                removed.append(_dump(node))
                return None
            return node
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in ('alignments', 'guard_losses')):
                target = call.func.value.id
                assert call.func.attr in ('append', 'extend') and len(call.args) == 1 and not call.keywords
                if target == 'alignments':
                    assert call.func.attr == 'append' and isinstance(call.args[0], ast.Call)
                    assert isinstance(call.args[0].func, ast.Name) and call.args[0].func.id == '_alignment'
                counts[target] += 1
                removed.append(_dump(node))
                return None
            return node
    # Clone through source AST, then remove only the allowlisted statements.
    prefix_tree = ast.Module(body=original_prefix, type_ignores=[])
    transformed = RemoveAuxiliary().visit(prefix_tree)
    assert counts == dict(alignments=4, guard_losses=4, initializers=2)
    assert not any(isinstance(node, ast.Name) and node.id in ('alignments', 'guard_losses')
                   for node in ast.walk(transformed))
    retained_pairs = [(key, value) for key, value in zip(returned.value.keys, returned.value.values)
                      if key.value in RETAINED_OUTPUTS]
    function.body = transformed.body + [ast.Return(value=ast.Dict(
        keys=[key for key, _ in retained_pairs], values=[value for _, value in retained_pairs]))]
    function.decorator_list = []
    receipt = dict(
        removed_statement_counts=counts,
        removed_auxiliary_statements_AST_sha256=hashlib.sha256('\n'.join(removed).encode()).hexdigest(),
        retained_score_prefix_AST_sha256=hashlib.sha256(_dump(transformed).encode()).hexdigest(),
        retained_outputs=list(RETAINED_OUTPUTS), omitted_outputs=list(OMITTED_OUTPUTS),
        score_math_unchanged=True, complete_auxiliary_dictionary_unchanged=False,
    )
    return ast.fix_missing_locations(tree), receipt


def router_score_AST(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef) and function.name == '_standardized_router'
    returned = function.body[-1]
    assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Dict)
    original_prefix = _dump(ast.Module(body=function.body[:-1], type_ignores=[]))
    pairs = list(zip(returned.value.keys, returned.value.values))
    removed = [(key, value) for key, value in pairs if key is None]
    assert len(removed) == 1
    call = removed[0][1]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == '_slotwise_routing_statistics' and not call.keywords
    assert [arg.id for arg in call.args] == ['probabilities', 'selected']
    kept = [(key, value) for key, value in pairs if key is not None]
    assert {key.value for key, _ in kept} == {'probabilities', 'weights', 'selected_mask', 'selected_indices'}
    returned.value = ast.Dict(keys=[key for key, _ in kept], values=[value for _, value in kept])
    assert _dump(ast.Module(body=function.body[:-1], type_ignores=[])) == original_prefix
    function.decorator_list = []
    return ast.fix_missing_locations(tree), dict(
        scoring_router_prefix_AST_sha256=hashlib.sha256(original_prefix.encode()).hexdigest(),
        probabilities_softmax_top2_weights_unchanged=True,
        omitted_only_slotwise_auxiliary_statistics=True,
    )


def install_score_only(model):
    original_forward = model.forward
    module = inspect.getmodule(type(model).forward)
    original_router = module._standardized_router
    forward_tree, forward_receipt = score_forward_AST(inspect.getsource(original_forward))
    router_tree, router_receipt = router_score_AST(inspect.getsource(original_router))
    forward_namespace = dict(original_forward.__func__.__globals__)
    router_namespace = dict(original_router.__globals__)
    exec(compile(forward_tree, inspect.getsourcefile(original_forward)+'::score_only', 'exec'), forward_namespace)
    exec(compile(router_tree, inspect.getsourcefile(original_router)+'::score_only', 'exec'), router_namespace)
    replacement = forward_namespace['forward']
    replacement.__defaults__ = original_forward.__func__.__defaults__
    replacement.__kwdefaults__ = original_forward.__func__.__kwdefaults__
    functools.update_wrapper(replacement, original_forward)
    model.forward = MethodType(replacement, model)
    module._standardized_router = router_namespace['_standardized_router']
    def restore():
        model.forward = original_forward
        module._standardized_router = original_router
    return restore, dict(forward=forward_receipt, router=router_receipt,
                         fusion_four_diagnostic_tensors_retained=True,
                         all_actual_neural_and_optical_operations_retained=True)


def _source_map():
    paths = (Path(__file__), Path(lowlevel.__file__), Path(projection.__file__))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _label_new_outputs(folder):
    # Both during execution and after the reused worker's final writes, make
    # this new leaf's narrowed auxiliary-output contract explicit.
    for filename in ('projection_identity.json', 'lowlevel_capture_identity.json', 'timing.json'):
        path = folder/filename
        if not path.exists():
            continue
        output = json.loads(path.read_text(encoding='utf-8'))
        output.pop('original_model_math_unchanged', None)
        output.pop('original_scopes_and_math_unchanged', None)
        output.pop('native_math_and_all_electronic_routes_retained', None)
        output.update(score_math_unchanged=True, auxiliary_outputs_omitted=True,
                      auxiliary_outputs_changed=True,
                      complete_auxiliary_dictionary_unchanged=False,
                      prediction_math_and_all_electronic_routes_retained=True,
                      fusion_four_diagnostic_tensors_retained=True)
        path.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')


def worker(args, guard, N):
    frozen = _source_map()
    path = args.output/'score_only_identity.json'
    identity = dict(
        status='running', profile='score_only_on_validated_lowlevel_capture_20261026',
        sources=frozen, video_calls=N, score_math_unchanged=True,
        auxiliary_outputs_omitted=True, complete_auxiliary_dictionary_unchanged=False,
        omitted_outputs=list(OMITTED_OUTPUTS), fusion_four_diagnostic_tensors_retained=True,
        original_2048_sealed_prediction_reference_required=True,
    )
    def save():
        path.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
    save()
    original_install = projection.install_simulation_boundaries
    def scoped_install(model, tape):
        restore_score, receipt = install_score_only(model)
        try:
            restore_boundary, ledger = ORIGINAL_INSTALL(model, tape)
        except BaseException:
            restore_score()
            raise
        identity['AST_receipt'] = receipt
        save()
        _label_new_outputs(args.output)
        def restore():
            try:
                restore_boundary()
            finally:
                restore_score()
        return restore, ledger + [dict(scope='score_only_inference', **receipt)]
    projection.install_simulation_boundaries = scoped_install
    try:
        result = lowlevel.worker(args, guard, N)
        if _source_map() != frozen:
            raise RuntimeError('Score-only candidate/original source changed')
        # Only the newly generated leaf is relabelled; no old raw is touched.
        _label_new_outputs(args.output)
        identity.update(status='complete', sources_unchanged=True,
                        original_2048_prediction_reference_bitwise_passed=True)
        save()
        return result
    except BaseException as error:
        identity.update(status='failed', error=repr(error), sources_unchanged=(_source_map()==frozen))
        save()
        raise
    finally:
        projection.install_simulation_boundaries = original_install
