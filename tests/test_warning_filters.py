"""Every ignore filter in the tests names the message it ignores.

The suite turns ``PointingWarning`` into an error (``filterwarnings`` in
``pyproject.toml``). A test whose subject is an advisory asserts it with
``pytest.warns``; a test that meets one incidentally carries a
``pytest.mark.filterwarnings("ignore:<message>:<category>")`` mark naming the
message, and a fixture or a shared helper, which cannot take a mark, wraps the
call in ``warnings.catch_warnings()`` with a ``warnings.filterwarnings`` naming
the message and the category. An ignore that names no message hides every
advisory that reaches the code it wraps, the ones nobody has met yet included.

The guard reads every Python file under ``tests/`` and fails on two kinds of
ignore filter:

* a blanket ignore, which names neither a message nor a category, such as
  ``warnings.simplefilter("ignore")`` or ``pytest.mark.filterwarnings("ignore")``:
  no file may hold one;
* a category-only ignore, which names a category but no message, such as
  ``warnings.simplefilter("ignore", PointingWarning)`` or
  ``pytest.mark.filterwarnings("ignore::<module>.<Category>")``, unless its file
  and category are in :data:`ALLOWED` with the reason the file may.

It reads ``warnings.simplefilter``, ``warnings.filterwarnings`` and
``warnings.catch_warnings(action=...)`` under any import alias, and their
arguments, like those of a ``filterwarnings`` mark, through the module-level
string constants they name; a mark argument it cannot read (an f-string, a
concatenation) counts as naming nothing. A message pattern that matches every
real message, such as ``".*"``, ``"."`` or ``"[A-Za-z]"`` (one that matches the
empty string, a lone NUL character, or both a lone ``a`` and a lone ``Z``),
names no message, and ``Warning`` names no category.
The guard also fails on an entry of the list that nothing matches any more, so
the list grows and shrinks only by a conscious edit.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

#: The functions of the ``warnings`` module that install a filter, each with
#: the parameters it takes by position (``catch_warnings`` takes keywords only).
_FILTER_FUNCTIONS = {
    "simplefilter": ("action", "category"),
    "filterwarnings": ("action", "message", "category"),
    "catch_warnings": (),
}

#: The category-only ignores the suite keeps on purpose, as
#: ``(file, category): reason``.
ALLOWED: dict[tuple[str, str], str] = {
    ("tests/test_source_ces_golden.py", "ResourceWarning"): (
        "garbage collection closes a file an earlier test left open, inside whichever test runs"
    ),
}


def _relative(path: Path) -> str:
    """Name a file by its path from the repository root."""
    return path.relative_to(_ROOT).as_posix()


def _test_files() -> list[Path]:
    """Every Python file under ``tests/``."""
    return sorted(_TESTS.rglob("*.py"))


def _string(node: ast.expr | None) -> str | None:
    """Return the value of a string literal, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _names_no_message(pattern: str | None) -> bool:
    r"""Report whether a message pattern is absent or matches every message.

    The pattern is a regular expression matched at the start of a message, so
    one that matches the empty string matches every message, and one that
    matches a lone NUL character, which no message holds, does so through a
    wildcard (``"."``, ``"\S"``) that matches nearly every real one. One that
    matches both a lone ``a`` and a lone ``Z`` is taken the same way: in
    practice it is a letter class (``"\w"``, ``"[A-Za-z]"``) that matches
    every message starting with a letter.
    """
    if pattern is None:
        return True
    try:
        return (
            re.match(pattern, "") is not None
            or re.match(pattern, "\x00") is not None
            or all(re.match(pattern, c) for c in "aZ")
        )
    except re.error:
        return False


def _category_name(name: str) -> str | None:
    """Name a category by the last part of its dotted name; None when it names none."""
    last = name.rpartition(".")[2].strip()
    return None if last in ("", "Warning") else last


def _warnings_bindings(tree: ast.AST) -> tuple[set[str], dict[str, str]]:
    """Return the names bound to the ``warnings`` module and to its filter functions."""
    modules: set[str] = set()
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == "warnings")
        elif isinstance(node, ast.ImportFrom) and node.module == "warnings" and not node.level:
            functions.update(
                {a.asname or a.name: a.name for a in node.names if a.name in _FILTER_FUNCTIONS}
            )
    return modules, functions


def _strings(node: ast.expr, constants: dict[str, list[str]]) -> list[str] | None:
    """Return the strings an expression holds, through ``constants``; None if any is not one."""
    if isinstance(node, ast.Starred):
        return _strings(node.value, constants)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, (ast.Tuple, ast.List)):
        parts = [_strings(element, constants) for element in node.elts]
        if any(part is None for part in parts):
            return None
        return [text for part in parts for text in part]
    text = _string(node)
    return None if text is None else [text]


def _text(node: ast.expr | None, constants: dict[str, list[str]]) -> str | None:
    """Return the one string an argument holds, through ``constants``; None otherwise."""
    texts = None if node is None else _strings(node, constants)
    return texts[0] if texts is not None and len(texts) == 1 else None


def _module_strings(tree: ast.Module) -> dict[str, list[str]]:
    """Map each module-level name bound to a string, or a tuple or list of them, to its strings."""
    constants: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        strings = _strings(value, constants)
        for target in targets:
            if isinstance(target, ast.Name) and strings is not None:
                constants[target.id] = strings
    return constants


def _is_filterwarnings_mark(func: ast.expr) -> bool:
    """Report whether a call's function is the ``filterwarnings`` mark."""
    if not (isinstance(func, ast.Attribute) and func.attr == "filterwarnings"):
        return False
    owner = func.value
    return (isinstance(owner, ast.Attribute) and owner.attr == "mark") or (
        isinstance(owner, ast.Name) and owner.id == "mark"
    )


def _filter_function(func: ast.expr, modules: set[str], functions: dict[str, str]) -> str | None:
    """Name the ``warnings`` filter function a call's function is, else None."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in modules and func.attr in _FILTER_FUNCTIONS:
            return func.attr
        return None
    if isinstance(func, ast.Name):
        return functions.get(func.id)
    return None


def _argument(call: ast.Call, function: str, name: str) -> ast.expr | None:
    """Return the argument a filter call passes for parameter ``name``, else None."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    positional = _FILTER_FUNCTIONS[function]
    if name in positional and positional.index(name) < len(call.args):
        return call.args[positional.index(name)]
    return None


def unnamed_ignores(source: str) -> list[tuple[int, str | None]]:
    """List the ignore filters in a Python source that name no message.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    list of (int, str or None)
        The line of each such filter and the category it names, None for a
        blanket ignore, in line order. A call whose message is neither a
        string literal nor a module-level string constant is taken to name
        one; a mark argument that is neither counts as a blanket ignore.
    """
    tree = ast.parse(source)
    modules, functions = _warnings_bindings(tree)
    constants = _module_strings(tree)
    found: list[tuple[int, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_filterwarnings_mark(node.func):
            for argument in node.args:
                specs = _strings(argument, constants)
                if specs is None:
                    found.append((node.lineno, None))
                    continue
                for spec in specs:
                    action, message, category, *_ = [*spec.split(":"), "", ""]
                    if action.strip() == "ignore" and _names_no_message(message.strip()):
                        found.append((node.lineno, _category_name(category)))
            continue
        function = _filter_function(node.func, modules, functions)
        if function is None or any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        if _text(_argument(node, function, "action"), constants) != "ignore":
            continue
        message = _argument(node, function, "message")
        text = _text(message, constants)
        if message is not None and text is None:
            continue
        if not _names_no_message(text):
            continue
        category = _argument(node, function, "category")
        found.append(
            (node.lineno, None if category is None else _category_name(ast.unparse(category)))
        )
    return sorted(found, key=lambda item: item[0])


@pytest.mark.parametrize("path", _test_files(), ids=_relative)
def test_no_test_file_ignores_every_warning(path: Path):
    """Every ignore filter in a test file names a message or, failing that, a category."""
    lines = [
        line for line, category in unnamed_ignores(path.read_text(encoding="utf-8")) if not category
    ]
    assert not lines, (
        f"{_relative(path)} ignores every warning at line(s) {lines}; name the message with a "
        "filterwarnings mark on the test, or a named filter in a fixture or helper"
    )


def _category_only_ignores() -> dict[tuple[str, str], list[int]]:
    """Every category-only ignore under ``tests/``, as ``(file, category): lines``."""
    found: dict[tuple[str, str], list[int]] = {}
    for path in _test_files():
        for line, category in unnamed_ignores(path.read_text(encoding="utf-8")):
            if category:
                found.setdefault((_relative(path), category), []).append(line)
    return found


def test_category_only_ignores_match_the_allowed_list():
    """The ignores that name a category but no message are exactly the listed ones."""
    observed = _category_only_ignores()
    new = {key: lines for key, lines in sorted(observed.items()) if key not in ALLOWED}
    stale = sorted(ALLOWED.keys() - observed.keys())
    assert not new and not stale, f"new category-only ignores: {new}\nstale entries: {stale}"


def test_the_scan_covers_the_whole_tests_tree():
    """The walk reaches nested directories, their conftest files and this file."""
    files = {_relative(path) for path in _test_files()}
    assert {
        "tests/conftest.py",
        "tests/overhead/calibration_night/conftest.py",
        "tests/visualization/test_source_track_plotting.py",
        _relative(Path(__file__).resolve()),
    } <= files


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "import warnings\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore')\n",
            [(3, None)],
            id="the-copied-idiom",
        ),
        pytest.param(
            "import warnings as w\nw.filterwarnings(action='ignore')\n",
            [(2, None)],
            id="module-alias-and-keyword",
        ),
        pytest.param(
            "from warnings import simplefilter as quiet\nquiet('ignore')\n",
            [(2, None)],
            id="imported-function",
        ),
        pytest.param(
            "import warnings\nwarnings.filterwarnings('ignore', '.*', Warning)\n",
            [(2, None)],
            id="a-message-matching-everything-and-category-warning",
        ),
        pytest.param(
            "import warnings\n"
            "import pytest\n"
            "warnings.filterwarnings('ignore', '.', PointingWarning)\n"
            "@pytest.mark.filterwarnings('ignore:\\\\S:a.PointingWarning')\n"
            "def test(): pass\n",
            [(3, "PointingWarning"), (4, "PointingWarning")],
            id="a-message-matching-every-real-message",
        ),
        pytest.param(
            "import warnings\n"
            "import pytest\n"
            "warnings.filterwarnings('ignore', '\\\\w', PointingWarning)\n"
            "@pytest.mark.filterwarnings('ignore:[A-Za-z]:a.PointingWarning')\n"
            "def test(): pass\n",
            [(3, "PointingWarning"), (4, "PointingWarning")],
            id="a-message-matching-every-message-that-starts-with-a-letter",
        ),
        pytest.param(
            "import warnings\n"
            "_IGNORE = 'ignore'\n"
            "_ANY = '.*'\n"
            "warnings.simplefilter(_IGNORE)\n"
            "warnings.filterwarnings(_IGNORE, _ANY, PointingWarning)\n",
            [(4, None), (5, "PointingWarning")],
            id="action-and-message-through-constants",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.filterwarnings(f'ignore:{TEXT}')\ndef test(): pass\n",
            [(2, None)],
            id="a-mark-the-scan-cannot-read",
        ),
        pytest.param(
            "import warnings\nwarnings.catch_warnings(action='ignore')\n",
            [(2, None)],
            id="catch-warnings-action",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.filterwarnings('ignore')\ndef test(): pass\n",
            [(2, None)],
            id="blanket-mark",
        ),
        pytest.param(
            "import warnings\nwarnings.simplefilter('ignore', PointingWarning)\n",
            [(2, "PointingWarning")],
            id="category-only-call",
        ),
        pytest.param(
            "import warnings\n"
            "warnings.filterwarnings('ignore', category=exceptions.PointingWarning)\n",
            [(2, "PointingWarning")],
            id="category-only-keyword",
        ),
        pytest.param(
            "import pytest\n"
            "_SPECS = ('ignore:High elevation:a.PointingWarning', 'ignore::a.PointingWarning')\n"
            "pytestmark = pytest.mark.filterwarnings(*_SPECS)\n",
            [(3, "PointingWarning")],
            id="category-only-mark-through-a-constant",
        ),
        pytest.param(
            "import warnings\n"
            "import numpy as np\n"
            "import pytest\n"
            "warnings.filterwarnings('ignore', message='High elevation', category=W)\n"
            "warnings.simplefilter('always')\n"
            "warnings.simplefilter('error', PointingWarning)\n"
            "np.seterr(all='ignore')\n"
            "@pytest.mark.filterwarnings('ignore:High elevation reduces:a.PointingWarning')\n"
            "@pytest.mark.filterwarnings('error::a.PointingWarning')\n"
            "def test(): pass\n",
            [],
            id="named-and-unrelated",
        ),
    ],
)
def test_the_scan_reads_an_ignore(source: str, expected: list[tuple[int, str | None]]):
    """The reader flags exactly the ignores that name no message."""
    assert unnamed_ignores(source) == expected
