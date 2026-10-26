"""Render the constant prompt string once INSIDE the first real timed field.

No tokenizer result, token tensor, feature or score is cached. The exact
original tokenizer still executes every field. Bulk preparation is unaffected.
Only the worker-local tokenize function's fixed string-template call changes.
"""
import ast
import time


class RenderOnce:
    def __init__(self):self.value=None;self.requests=0;self.renders=0;self.first_interval=None
    def __call__(self,render):
        self.requests+=1
        if self.value is None:
            a=time.perf_counter();value=render();b=time.perf_counter()
            if not isinstance(value,str):raise ValueError('Only fixed prompt TEXT may be retained')
            self.value=value;self.renders+=1;self.first_interval=[a,b]
        return self.value


def prompt_render_AST(tree):
    count=0
    class RenderStringOnly(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            nonlocal count
            if node.name!='tokenize':return self.generic_visit(node)
            for item in node.body:
                if (isinstance(item,ast.Assign) and len(item.targets)==1
                        and isinstance(item.targets[0],ast.Name) and item.targets[0].id=='rendered'):
                    call=item.value
                    if (not isinstance(call,ast.Call) or ast.unparse(call.func)!='text.render_prompt'
                            or len(call.args)!=1 or ast.unparse(call.args[0])!='front.tokenizer'
                            or {k.arg:ast.unparse(k.value) for k in call.keywords}!=
                            {'target_name':"'temporal'",'prompt':"text.TARGET_PROMPTS['temporal']"}):
                        raise ValueError('Exact constant temporal prompt call required')
                    count+=1
                    item.value=ast.Call(func=ast.Name(id='development_render_once',ctx=ast.Load()),
                        args=[ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],
                            kw_defaults=[],defaults=[]),body=call)],keywords=[])
            return node
    tree=RenderStringOnly().visit(tree)
    if count!=1:raise ValueError('Exactly one fixed worker-local prompt-render call required')
    return ast.fix_missing_locations(tree)
