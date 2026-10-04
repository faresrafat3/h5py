"""Check that every public method which raises documents what it raises.

gh-1447: exceptions were not documented anywhere. A behaviour test does not
catch that — it passes whether or not the docstring names the exception. This
check fails when a raising public method's docstring is silent.

The work happens inside the test function, not at collection time. A
module-level parametrize list freezes the result at import, so the check can
report a stale answer; reading the source at run time avoids that.

Run as part of the test suite:
    pytest h5py/tests/test_hl_exception_docs.py
"""

import ast
import inspect
import pathlib

import pytest

import h5py

# The public classes. The list mirrors h5py/__init__.py: anything else in _hl
# is internal, and h5py/_hl/__init__.py says so explicitly.
PUBLIC_CLASSES = (
    h5py.File,
    h5py.Group,
    h5py.Dataset,
    h5py.Datatype,
    h5py.AttributeManager,
    h5py.VirtualSource,
    h5py.VirtualLayout,
    h5py.HLObject,
)

# Exception class names that must appear in a docstring for it to count.
# The word "raise" alone is NOT enough: gh-1447 exists precisely because the
# docs said "will raise an exception" without naming the class.
_EXC_CLASSES = (
    "TypeError", "ValueError", "KeyError", "IndexError", "OSError",
    "RuntimeError", "OverflowError", "StopIteration",
    "NotImplementedError", "FileNotFoundError", "FileExistsError",
    "AttributeError", "AssertionError", "ImportError",
)


def _raises_in(node: ast.AST) -> list[ast.Raise]:
    """Raise statements belonging to this function, not to nested functions."""
    found = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # a nested function's raises are its own
        if isinstance(child, ast.Raise):
            found.append(child)
        found.extend(_raises_in(child))
    return found


def _silences_missing_docstring(node: ast.AST, src_lines: list[str]) -> bool:
    """True when the function body carries `# pylint: disable=missing-docstring`.

    A property setter is the common case: the docstring lives on the getter,
    and the setter is marked as intentionally undocumented.
    """
    start = node.lineno - 1
    end = getattr(node, "end_lineno", start + 1)
    header = "\n".join(src_lines[start:end])
    return "pylint: disable=missing-docstring" in header


def _docstring_of(node: ast.AST) -> str:
    body = getattr(node, "body", None) or []
    if body and isinstance(body[0], ast.Expr):
        v = body[0].value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
    return ""


def _public_raising_methods() -> list[tuple[str, str, int, bool]]:
    """(class, method, raise count, documents an exception), read fresh."""
    out = []
    for cls in PUBLIC_CLASSES:
        try:
            src_file = inspect.getsourcefile(cls)
            src = pathlib.Path(src_file).read_text()
        except (TypeError, OSError):
            continue
        tree = ast.parse(src)
        src_lines = src.splitlines()
        for cnode in [n for n in tree.body
                      if isinstance(n, ast.ClassDef) and n.name == cls.__name__]:
            # Docstrings of property getters, so a skipped setter can be
            # checked against the docstring that covers the property.
            getter_docs = {
                m.name: _docstring_of(m)
                for m in cnode.body
                if isinstance(m, ast.FunctionDef)
                and any(ast.unparse(d) == "property" for d in m.decorator_list)
            }
            for m in cnode.body:
                if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if m.name.startswith("_") and m.name != "__init__":
                    continue
                raises = _raises_in(m)
                if not raises:
                    continue
                ds = _docstring_of(m)
                if _silences_missing_docstring(m, src_lines):
                    # The property's docstring lives on the getter and must
                    # document the setter's raise; otherwise nobody does.
                    ds = getter_docs.get(m.name, "")
                documents = any(t in ds for t in _EXC_CLASSES)
                out.append((cls.__name__, m.name, len(raises), documents))
    return out


def test_public_raising_method_documents_the_exception():
    """A public method that can raise must name the exception in its docstring."""
    methods = _public_raising_methods()
    assert methods, "found no raising public methods — the check is broken"

    silent = [(c, m, n) for c, m, n, documented in methods if not documented]
    if silent:
        listing = "\n".join(
            f"  {c}.{m}  ({n} raise site(s))" for c, m, n in silent
        )
        pytest.fail(
            f"{len(silent)} of {len(methods)} public methods raise but do not "
            f"name an exception in their docstring.\n"
            f"gh-1447: a caller cannot write a targeted try/except without "
            f"knowing the type.\n{listing}"
        )


def test_the_check_sees_the_known_raising_methods():
    """Guard against the check silently finding nothing."""
    methods = _public_raising_methods()
    assert len(methods) >= 15, f"only found {len(methods)} raising public methods"
    names = {m for _, m, _, _ in methods}
    # These are the methods gh-1447 was about; they must be visible.
    for expected in ("read_direct", "resize", "create", "get", "copy"):
        assert expected in names, f"{expected} missing from the check"
