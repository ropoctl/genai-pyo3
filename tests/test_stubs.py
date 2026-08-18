"""The type stubs must describe the module that actually ships.

Stubs drift silently: nothing at runtime notices when a stub documents a
constructor that no longer exists, or omits a type that does. These checks
are cheap and catch exactly that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import genai_pyo3 as g

STUB_PATH = Path(g.__file__).with_name("__init__.pyi")


def stub_tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text())


def stub_all() -> list[str]:
    for node in stub_tree().body:
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "__all__" for t in node.targets)
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("no __all__ in the stub")


def stub_toplevel_names() -> set[str]:
    names: set[str] = set()
    for node in stub_tree().body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_stub_all_matches_package_all():
    assert sorted(stub_all()) == sorted(g.__all__)


def test_every_exported_name_is_defined_in_the_stub():
    missing = sorted(set(g.__all__) - stub_toplevel_names())
    assert not missing, f"exported but undocumented in the stub: {missing}"


def test_stub_documents_no_removed_names():
    # Anything the stub declares should exist at runtime.
    declared = {name for name in stub_toplevel_names() if not name.startswith("_")}
    phantom = sorted(name for name in declared if not hasattr(g, name))
    assert not phantom, f"documented in the stub but missing at runtime: {phantom}"


def test_removed_constructors_are_really_gone():
    for name in (
        "with_api_key",
        "with_api_key_and_base_url",
        "with_base_url",
        "with_request_override",
    ):
        assert not hasattr(g.Client, name), f"Client.{name} should have been removed"
    assert "with_api_key" not in STUB_PATH.read_text()


def test_public_types_are_importable_from_types_module():
    from genai_pyo3 import types

    for name in types.__all__:
        assert hasattr(types, name)
        assert hasattr(g, name), f"{name} in types.__all__ but not re-exported"
