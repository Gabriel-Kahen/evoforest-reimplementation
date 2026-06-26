from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np

from evoforest_arch.graph import CallableFamily, EvalContext, FeatureBlock, NodeAlternative, ResidualWeightRule


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


def build_source_alternative(
    alternative_id: str,
    parents: tuple[str, ...],
    source: str,
    *,
    description: str = "",
    global_refs: tuple[str, ...] = (),
    tags: tuple[str, ...] = ("source",),
) -> NodeAlternative:
    """Compile a trusted lambda into a graph alternative.

    The paper's mutation prompt represents alternatives as Python lambdas. This
    helper recreates that source-backed path for local, trusted experiments. It is
    not a security sandbox.
    """

    callable_obj = compile_lambda_source(source)

    def fn(ctx: EvalContext, values: dict[str, Any]) -> Any:
        return callable_obj(ctx, values)

    return NodeAlternative(
        id=alternative_id,
        parents=parents,
        fn=fn,
        description=description or f"Source-backed alternative {alternative_id}.",
        tags=tags,
        primitive="source",
        global_refs=global_refs,
        source=source.strip(),
    )


def compile_lambda_source(source: str) -> object:
    expr = source.strip()
    if not expr.startswith("lambda "):
        raise ValueError("Source alternatives must be Python lambda expressions accepting (ctx, values).")
    _validate_lambda_ast(expr)
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "math": math,
        "FeatureBlock": FeatureBlock,
        "CallableFamily": CallableFamily,
        "ResidualWeightRule": ResidualWeightRule,
    }
    compiled = eval(expr, namespace, {})  # noqa: S307 - trusted local mutation source, not a sandbox.
    if not callable(compiled):
        raise TypeError("Source alternative did not evaluate to a callable.")
    return compiled


def _validate_lambda_ast(source: str) -> None:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid source lambda syntax: {exc}") from exc
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError("Source alternatives must parse to one Python lambda expression.")
    args = [arg.arg for arg in tree.body.args.args]
    if args[:2] != ["ctx", "values"]:
        raise ValueError("Source lambda must accept ctx and values as its first two arguments.")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError(f"Unsupported source lambda syntax: {type(node).__name__}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Source lambda may not access dunder attributes.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Source lambda may not access dunder names.")
