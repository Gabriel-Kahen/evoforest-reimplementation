from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import multiprocessing as mp
import pathlib
import queue
import sys
from typing import Any

import numpy as np

from evoforest_arch.graph import CallableFamily, EvalContext, FeatureBlock, NodeAlternative, ResidualWeightRule
from evoforest_arch.globals import GlobalStore


SAFE_BUILTINS: dict[str, object] = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "slice": slice,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass(frozen=True)
class SourceSandboxPolicy:
    enabled: bool = True
    timeout_seconds: float = 2.0
    memory_mb: int = 512
    cpu_seconds: int = 2
    deterministic_trials: int = 2


class SourceExecutionError(RuntimeError):
    pass


class SourceTimeoutError(SourceExecutionError):
    pass


def build_source_alternative(
    alternative_id: str,
    parents: tuple[str, ...],
    source: str,
    *,
    description: str = "",
    global_refs: tuple[str, ...] = (),
    tags: tuple[str, ...] = ("source",),
    node_kind: str = "",
    output_contract: dict[str, object] | None = None,
    torch_source: str = "",
    sandbox_policy: SourceSandboxPolicy | None = None,
) -> NodeAlternative:
    """Compile a source-backed graph alternative with sandboxed execution."""

    source = source.strip()
    torch_source = torch_source.strip()
    _validate_lambda_ast(source)
    explicit_torch_source = bool(torch_source)
    if not torch_source and node_kind in {"", "intermediate", "output"}:
        torch_source = _derive_torch_source(source)
    if torch_source:
        _validate_lambda_ast(torch_source, backend="torch")
    policy = sandbox_policy or SourceSandboxPolicy()
    contract = dict(output_contract or {})
    if bool(contract.get("differentiable", False)) and not torch_source:
        raise ValueError("Source alternatives with output_contract.differentiable=true require torch_source.")
    if torch_source:
        contract.setdefault("differentiable", True)
        contract.setdefault("differentiability_source", "provided" if explicit_torch_source else "auto")
    else:
        contract.setdefault("differentiable", False)
    tags = tuple(dict.fromkeys((*tags, "sandboxed" if policy.enabled else "unsandboxed")))
    local_callable: object | None = None
    torch_callable: object | None = None

    def fn(ctx: EvalContext, values: dict[str, Any]) -> Any:
        nonlocal local_callable
        if policy.enabled:
            result = _run_sandboxed_trials(source, ctx, values, node_kind, contract, policy)
        else:
            if local_callable is None:
                local_callable = compile_lambda_source(source)
            result = _materialize_source_result(local_callable(ctx, values), node_kind=node_kind, output_contract=contract)  # type: ignore[misc]
            _validate_source_result(result, node_kind=node_kind, output_contract=contract, sample_count=_infer_sample_count(ctx.inputs))
        return result

    def tfn(ctx: EvalContext, values: dict[str, Any]) -> Any:
        nonlocal torch_callable
        if not torch_source:
            raise SourceExecutionError(f"Source alternative {alternative_id!r} has no torch_source.")
        if torch_callable is None:
            torch_callable = compile_torch_lambda_source(torch_source)
        return torch_callable(ctx, values)  # type: ignore[misc]

    return NodeAlternative(
        id=alternative_id,
        parents=parents,
        fn=fn,
        description=description or f"Source-backed alternative {alternative_id}.",
        tags=tags,
        primitive="source",
        global_refs=global_refs,
        torch_fn=tfn if torch_source else None,
        source=source,
        torch_source=torch_source,
        output_contract=contract,
    )


def compile_lambda_source(source: str) -> object:
    expr = source.strip()
    if not expr.startswith("lambda "):
        raise ValueError("Source alternatives must be Python lambda expressions accepting (ctx, values).")
    _validate_lambda_ast(expr)
    namespace = _source_namespace()
    compiled = eval(expr, namespace, {})  # noqa: S307 - expression is validated and runtime is sandboxed by caller.
    if not callable(compiled):
        raise TypeError("Source alternative did not evaluate to a callable.")
    return compiled


def compile_torch_lambda_source(source: str) -> object:
    expr = source.strip()
    if not expr.startswith("lambda "):
        raise ValueError("Torch source alternatives must be Python lambda expressions accepting (ctx, values).")
    _validate_lambda_ast(expr, backend="torch")
    import torch

    namespace = _source_namespace()
    namespace["torch"] = torch
    compiled = eval(expr, namespace, {})  # noqa: S307 - expression is validated and used only in requested torch refinement.
    if not callable(compiled):
        raise TypeError("Torch source alternative did not evaluate to a callable.")
    return compiled


def _derive_torch_source(source: str) -> str:
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return ""
    if not isinstance(tree.body, ast.Lambda):
        return ""
    transformed = _TorchSourceTransformer().visit(tree)
    ast.fix_missing_locations(transformed)
    try:
        candidate = ast.unparse(transformed)
        _validate_lambda_ast(candidate, backend="torch")
    except Exception:
        return ""
    return candidate


class _TorchSourceTransformer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "np":
            return ast.copy_location(ast.Name(id="torch", ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node, ast.Attribute) and node.attr == "values":
            return node.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr == "asarray":
            node.attr = "as_tensor"
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FeatureBlock":
            if node.args:
                return node.args[0]
        return node


def _source_namespace() -> dict[str, object]:
    return {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "math": math,
        "FeatureBlock": FeatureBlock,
        "CallableFamily": CallableFamily,
        "ResidualWeightRule": ResidualWeightRule,
    }


def _run_sandboxed_trials(
    source: str,
    ctx: EvalContext,
    values: dict[str, Any],
    node_kind: str,
    output_contract: dict[str, object],
    policy: SourceSandboxPolicy,
) -> Any:
    sample_count = _infer_sample_count(ctx.inputs)
    trials = max(1, int(policy.deterministic_trials))
    first = _materialize_source_result(
        _execute_source_in_subprocess(source, ctx.inputs, ctx.globals.clone(), values, policy),
        node_kind=node_kind,
        output_contract=output_contract,
    )
    _validate_source_result(first, node_kind=node_kind, output_contract=output_contract, sample_count=sample_count)
    for _index in range(1, trials):
        candidate = _materialize_source_result(
            _execute_source_in_subprocess(source, ctx.inputs, ctx.globals.clone(), values, policy),
            node_kind=node_kind,
            output_contract=output_contract,
        )
        _validate_source_result(candidate, node_kind=node_kind, output_contract=output_contract, sample_count=sample_count)
        if not _equivalent_result(first, candidate):
            raise SourceExecutionError("Source alternative produced non-deterministic outputs for identical inputs.")
    return first


def _execute_source_in_subprocess(
    source: str,
    inputs: dict[str, Any],
    globals_store: GlobalStore,
    values: dict[str, Any],
    policy: SourceSandboxPolicy,
) -> Any:
    try:
        context = mp.get_context(_sandbox_start_method())
        result_queue: mp.Queue[Any] = context.Queue(maxsize=1)
        process = context.Process(
            target=_source_worker,
            args=(result_queue, source, inputs, globals_store, values, int(policy.memory_mb), int(policy.cpu_seconds)),
        )
        process.start()
    except Exception as exc:
        raise SourceExecutionError(f"Could not start source sandbox process: {exc}") from exc

    process.join(max(0.01, float(policy.timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        if process.is_alive():
            process.kill()
            process.join(0.2)
        raise SourceTimeoutError(f"Source alternative exceeded {policy.timeout_seconds:.3f}s timeout.")

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise SourceExecutionError(f"Source sandbox exited without returning a result (exitcode={process.exitcode}).") from exc
    if status == "ok":
        return payload
    raise SourceExecutionError(str(payload))


def _source_worker(
    result_queue: Any,
    source: str,
    inputs: dict[str, Any],
    globals_store: GlobalStore,
    values: dict[str, Any],
    memory_mb: int,
    cpu_seconds: int,
) -> None:
    try:
        _apply_resource_limits(memory_mb, cpu_seconds)
        callable_obj = compile_lambda_source(source)
        ctx = EvalContext(inputs=inputs, globals=globals_store)
        result_queue.put(("ok", callable_obj(ctx, values)))
    except BaseException as exc:  # noqa: BLE001 - worker must report every failure to parent.
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _apply_resource_limits(memory_mb: int, cpu_seconds: int) -> None:
    try:
        import resource
        if memory_mb > 0:
            limit = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        if cpu_seconds > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 1))
    except Exception:
        return


def _sandbox_start_method() -> str:
    main_file = str(getattr(sys.modules.get("__main__"), "__file__", "") or "")
    if main_file and pathlib.Path(main_file).exists():
        return "spawn"
    if "fork" in mp.get_all_start_methods():
        return "fork"
    return "spawn"


def _materialize_source_result(result: Any, *, node_kind: str, output_contract: dict[str, object]) -> Any:
    if isinstance(result, dict):
        expected_type = str(output_contract.get("type", ""))
        kind = str(result.get("kind", expected_type))
        if node_kind == "callable" or kind == "callable":
            return _callable_family_from_spec(result)
        if expected_type == "residual_weight_rule" or kind in {"residual_weight_rule", "residual_weight"}:
            return _residual_weight_rule_from_spec(result)
    return result


def _callable_family_from_spec(spec: dict[str, Any]) -> CallableFamily:
    op = str(spec.get("op", "identity"))
    name = str(spec.get("name", op))
    scale = float(spec.get("scale", 1.0))
    low = float(spec.get("low", -1.0))
    high = float(spec.get("high", 1.0))

    def apply(x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float64)
        if op == "identity":
            return array
        if op == "sigmoid":
            return 1.0 / (1.0 + np.exp(-scale * array))
        if op == "tanh":
            return np.tanh(scale * array)
        if op == "clipped_linear":
            return np.clip(scale * array, low, high)
        raise SourceExecutionError(f"Unsupported callable source op {op!r}.")

    if op not in {"identity", "sigmoid", "tanh", "clipped_linear"}:
        raise SourceExecutionError(f"Unsupported callable source op {op!r}.")
    return CallableFamily(name=name, apply=apply, description=str(spec.get("description", f"Source callable {op}.")))


def _residual_weight_rule_from_spec(spec: dict[str, Any]) -> ResidualWeightRule:
    op = str(spec.get("op", "identity"))
    name = str(spec.get("name", op))
    scale = max(float(spec.get("scale", 1.0)), 1e-8)

    def apply(residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64)
        if op == "identity":
            return np.ones_like(r)
        if op == "huber":
            return np.minimum(1.0, scale / np.maximum(np.abs(r), 1e-8))
        if op == "soft_l1":
            return 1.0 / np.sqrt(1.0 + (r / scale) ** 2)
        raise SourceExecutionError(f"Unsupported residual-weight source op {op!r}.")

    if op not in {"identity", "huber", "soft_l1"}:
        raise SourceExecutionError(f"Unsupported residual-weight source op {op!r}.")
    return ResidualWeightRule(name=name, apply=apply, description=str(spec.get("description", f"Source residual-weight rule {op}.")))


def _validate_source_result(
    result: Any,
    *,
    node_kind: str,
    output_contract: dict[str, object],
    sample_count: int | None,
) -> None:
    expected_type = str(output_contract.get("type", ""))
    if expected_type == "callable" and not isinstance(result, CallableFamily):
        raise SourceExecutionError("Source output contract expected a CallableFamily.")
    if expected_type == "residual_weight_rule" and not isinstance(result, ResidualWeightRule):
        raise SourceExecutionError("Source output contract expected a ResidualWeightRule.")
    if node_kind == "callable" and not isinstance(result, CallableFamily):
        raise SourceExecutionError("Source alternative for a callable node must return CallableFamily.")

    if node_kind in {"intermediate", "output"} or expected_type in {"feature_block", "array"}:
        if isinstance(result, FeatureBlock):
            array = result.values
            names = result.names
        else:
            array = np.asarray(result, dtype=np.float64)
            names = []
        if array.ndim == 1:
            n_rows = int(array.shape[0])
            n_columns = 1
        elif array.ndim == 2:
            n_rows = int(array.shape[0])
            n_columns = int(array.shape[1])
        else:
            raise SourceExecutionError(f"Source feature output must be 1-D or 2-D, got shape {array.shape}.")
        if sample_count is not None and n_rows != sample_count:
            raise SourceExecutionError(f"Source feature output has {n_rows} rows, expected {sample_count}.")
        expected_columns = output_contract.get("n_columns")
        if expected_columns is not None and n_columns != int(expected_columns):
            raise SourceExecutionError(f"Source feature output has {n_columns} columns, expected {int(expected_columns)}.")
        min_columns = output_contract.get("min_columns")
        if min_columns is not None and n_columns < int(min_columns):
            raise SourceExecutionError(f"Source feature output has {n_columns} columns, expected at least {int(min_columns)}.")
        max_columns = output_contract.get("max_columns")
        if max_columns is not None and n_columns > int(max_columns):
            raise SourceExecutionError(f"Source feature output has {n_columns} columns, expected at most {int(max_columns)}.")
        if names and len(names) != n_columns:
            raise SourceExecutionError("Source FeatureBlock names do not match output columns.")


def _infer_sample_count(inputs: dict[str, Any]) -> int | None:
    for value in inputs.values():
        array = np.asarray(value)
        if array.ndim > 0:
            return int(array.shape[0])
    return None


def _equivalent_result(left: Any, right: Any) -> bool:
    if isinstance(left, FeatureBlock) or isinstance(right, FeatureBlock):
        if not isinstance(left, FeatureBlock) or not isinstance(right, FeatureBlock):
            return False
        return left.names == right.names and bool(np.allclose(left.values, right.values, equal_nan=True))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.allclose(np.asarray(left), np.asarray(right), equal_nan=True))
    if isinstance(left, (float, int, str, bool, type(None))):
        return left == right
    if isinstance(left, CallableFamily) and isinstance(right, CallableFamily):
        return left.name == right.name and left.description == right.description
    if isinstance(left, ResidualWeightRule) and isinstance(right, ResidualWeightRule):
        return left.name == right.name and left.description == right.description
    return repr(left) == repr(right)


def _validate_lambda_ast(source: str, *, backend: str = "numpy") -> None:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid source lambda syntax: {exc}") from exc
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError("Source alternatives must parse to one Python lambda expression.")
    lambda_node = tree.body
    args = [arg.arg for arg in lambda_node.args.args]
    if args[:2] != ["ctx", "values"]:
        raise ValueError("Source lambda must accept ctx and values as its first two arguments.")
    if len(args) != 2 or lambda_node.args.defaults or lambda_node.args.kw_defaults or lambda_node.args.vararg or lambda_node.args.kwarg:
        raise ValueError("Source lambda must accept exactly (ctx, values) with no defaults, varargs, or keyword-only arguments.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and node is not lambda_node:
            raise ValueError("Nested lambdas are not supported in source alternatives.")
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError(f"Unsupported source lambda syntax: {type(node).__name__}.")
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise ValueError(f"Unsupported source lambda syntax: {type(node).__name__}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Source lambda may not access dunder attributes.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Source lambda may not access dunder names.")
        if isinstance(node, ast.Call):
            _validate_call_node(node, backend=backend)
        if isinstance(node, ast.Attribute):
            _validate_attribute_node(node, backend=backend)
        if isinstance(node, ast.Name):
            _validate_name_node(node, backend=backend)


ALLOWED_NAMES = {
    "ctx",
    "values",
    "np",
    "math",
    "FeatureBlock",
    "float",
    "int",
    "len",
    "max",
    "min",
    "abs",
    "sum",
    "round",
    "list",
    "tuple",
    "bool",
}
ALLOWED_TORCH_NAMES = ALLOWED_NAMES | {"torch"}
ALLOWED_NUMPY_CALLS = {
    "abs",
    "asarray",
    "clip",
    "column_stack",
    "concatenate",
    "cumsum",
    "diff",
    "full",
    "linspace",
    "log",
    "maximum",
    "max",
    "mean",
    "median",
    "minimum",
    "min",
    "nan_to_num",
    "ones",
    "quantile",
    "reshape",
    "square",
    "stack",
    "std",
    "sum",
    "sort",
    "zeros",
}
ALLOWED_MATH_CALLS = {"ceil", "exp", "floor", "log", "sqrt"}
ALLOWED_TORCH_CALLS = {
    "abs",
    "as_tensor",
    "cat",
    "clamp",
    "column_stack",
    "diff",
    "log",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "reshape",
    "square",
    "stack",
    "std",
    "sum",
    "zeros_like",
    "ones_like",
}
ALLOWED_BUILTIN_CALLS = {"FeatureBlock", "float", "int", "len", "max", "min", "abs", "sum", "round", "list", "tuple", "bool"}
ALLOWED_DATA_ATTRIBUTES = {"values", "shape", "T"}


def _validate_name_node(node: ast.Name, *, backend: str) -> None:
    allowed = ALLOWED_TORCH_NAMES if backend == "torch" else ALLOWED_NAMES
    if node.id not in allowed:
        raise ValueError(f"Unsupported source name {node.id!r}.")


def _validate_attribute_node(node: ast.Attribute, *, backend: str) -> None:
    if isinstance(node.value, ast.Name):
        if node.value.id == "np" and node.attr in ALLOWED_NUMPY_CALLS:
            return
        if node.value.id == "math" and node.attr in ALLOWED_MATH_CALLS:
            return
        if backend == "torch" and node.value.id == "torch" and node.attr in ALLOWED_TORCH_CALLS:
            return
        if backend == "torch" and node.value.id == "torch" and node.attr in {"float32", "float64"}:
            return
    if isinstance(node.value, ast.Name) and node.value.id == "ctx" and node.attr in {"globals", "read_input"}:
        return
    if isinstance(node.value, ast.Name) and node.value.id == "values" and node.attr == "get":
        return
    if (
        isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "ctx"
        and node.value.attr == "globals"
        and node.attr == "get"
    ):
        return
    if node.attr in ALLOWED_DATA_ATTRIBUTES:
        return
    raise ValueError(f"Unsupported source attribute access .{node.attr}.")


def _validate_call_node(node: ast.Call, *, backend: str) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in ALLOWED_BUILTIN_CALLS:
            return
        raise ValueError(f"Unsupported source call {func.id}(...).")
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "np" and func.attr in ALLOWED_NUMPY_CALLS:
            return
        if isinstance(func.value, ast.Name) and func.value.id == "math" and func.attr in ALLOWED_MATH_CALLS:
            return
        if backend == "torch" and isinstance(func.value, ast.Name) and func.value.id == "torch" and func.attr in ALLOWED_TORCH_CALLS:
            return
        if isinstance(func.value, ast.Name) and func.value.id == "ctx" and func.attr == "read_input":
            return
        if (
            isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "ctx"
            and func.value.attr == "globals"
            and func.attr == "get"
        ):
            return
        if isinstance(func.value, ast.Name) and func.value.id == "values" and func.attr == "get":
            return
    raise ValueError("Unsupported source call target.")
