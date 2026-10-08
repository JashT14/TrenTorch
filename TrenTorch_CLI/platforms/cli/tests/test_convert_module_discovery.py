"""
MC/DC coverage for ConvertCommand.run's two module-discovery decisions:
the "all" filter (d.is_dir() and not d.name.startswith((".", "_"))) and
the specific-module 4-atom OR
(d.is_dir() and (d.name == args.module or d.name.startswith(f"{args.module}_")
 or d.name.endswith(f"_{args.module}"))).

The real conversion functions (to_qmd/to_ipynb/to_sandbox_code/to_platform_yaml)
are mocked out -- this file only exercises which module directories get
selected, not the conversion logic itself.
"""

from argparse import Namespace
from io import StringIO
from pathlib import Path

from rich.console import Console

from platforms.cli.core.config import CLIConfig
from platforms.cli.processes.convert import ConvertCommand

# Captured at import time so patching Path.iterdir twice cannot stack patches.
_REAL_ITERDIR = Path.iterdir


def _run_convert(tmp_path, monkeypatch, entries: dict, *, module="all", fmt="qmd", root=None):
    """entries: name -> "dir_with_src" | "dir_no_src" | "file"."""
    project_root = tmp_path if root is None else root
    src_dir = project_root / "data" / "src"
    src_dir.mkdir(parents=True)
    for name, kind in entries.items():
        if kind == "dir_with_src":
            (src_dir / name).mkdir()
            (src_dir / name / f"{name}.py").write_text("x = 1\n", encoding="utf-8")
        elif kind == "dir_no_src":
            (src_dir / name).mkdir()
        else:
            (src_dir / name).write_text("", encoding="utf-8")

    import trentorch.export_sanitizer as sanitizer_module

    monkeypatch.setattr(sanitizer_module, "to_qmd", lambda content: "qmd")
    monkeypatch.setattr(sanitizer_module, "to_ipynb", lambda content: {"cells": []})
    monkeypatch.setattr(sanitizer_module, "to_sandbox_code", lambda content: "code")
    monkeypatch.setattr(sanitizer_module, "to_platform_yaml", lambda content, module_name=None: "yaml: true")

    cmd = ConvertCommand(CLIConfig.from_project_root(project_root))
    buf = StringIO()
    cmd.console = Console(file=buf, width=200, no_color=True)
    result = cmd.run(Namespace(module=module, format=fmt, out=str(project_root / "build")))
    return result, buf.getvalue()


# ---------------------------------------------------------------------------
# "all" filter: d.is_dir() and not d.name.startswith((".", "_"))
# ---------------------------------------------------------------------------


def test_regular_module_directory_is_converted(tmp_path, monkeypatch):
    """Baseline: is_dir() True, name doesn't start with "." or "_" ->
    included."""
    _, out = _run_convert(tmp_path, monkeypatch, {"01_tensor": "dir_with_src"})
    assert "01_tensor" in out
    assert "Converting 1 module" in out


def test_dotfile_directory_is_excluded_from_all(tmp_path, monkeypatch):
    """is_dir() True, name starts with "." -> excluded. Paired with the
    baseline: only the leading-dot name differs, isolating that half of
    the and."""
    _, out = _run_convert(
        tmp_path,
        monkeypatch,
        {"01_tensor": "dir_with_src", ".hidden": "dir_with_src"},
    )
    assert "Converting 1 module" in out


def test_underscore_prefixed_directory_is_excluded_from_all(tmp_path, monkeypatch):
    """is_dir() True, name starts with "_" -> excluded. Paired with the
    baseline: only the leading-underscore name differs, isolating that
    half of the and (distinct atom from the dot-prefix test, since
    startswith((".", "_")) is itself a 2-way check inside the not)."""
    _, out = _run_convert(
        tmp_path,
        monkeypatch,
        {"01_tensor": "dir_with_src", "__pycache__": "dir_with_src"},
    )
    assert "Converting 1 module" in out


def test_plain_file_is_excluded_from_all(tmp_path, monkeypatch):
    """is_dir() False (a regular file) -> excluded regardless of name.
    Paired with the baseline: only is_dir()'s result differs, isolating
    that half of the and."""
    _, out = _run_convert(tmp_path, monkeypatch, {"01_tensor": "dir_with_src", "readme.txt": "file"})
    assert "Converting 1 module" in out


# ---------------------------------------------------------------------------
# specific-module OR: d.is_dir() and (name == module or
#                                      name.startswith(f"{module}_") or
#                                      name.endswith(f"_{module}"))
# ---------------------------------------------------------------------------


def test_exact_name_match_is_found(tmp_path, monkeypatch):
    """Baseline: is_dir() True, name == args.module -> found."""
    result, out = _run_convert(tmp_path, monkeypatch, {"tensor": "dir_with_src"}, module="tensor")
    assert result == 0
    assert "Module not found" not in out


def test_prefix_match_is_found(tmp_path, monkeypatch):
    """name == module False, name.startswith(f"{module}_") True ->
    found. Paired with the baseline: only which disjunct matches
    differs, isolating this atom."""
    result, out = _run_convert(tmp_path, monkeypatch, {"01_tensor": "dir_with_src"}, module="01")
    assert result == 0
    assert "Module not found" not in out


def test_suffix_match_is_found(tmp_path, monkeypatch):
    """Neither exact nor prefix match, name.endswith(f"_{module}") True
    -> found. Isolates the third disjunct."""
    result, out = _run_convert(tmp_path, monkeypatch, {"foo_tensor": "dir_with_src"}, module="tensor")
    assert result == 0
    assert "Module not found" not in out


def test_no_match_reports_module_not_found(tmp_path, monkeypatch):
    """All three disjuncts False -> not found, run() returns 1. Paired
    with the exact-match baseline: only the module name's relation to
    the directory name differs."""
    result, out = _run_convert(tmp_path, monkeypatch, {"01_tensor": "dir_with_src"}, module="unrelated")
    assert result == 1
    assert "Module not found: unrelated" in out


def test_file_named_exactly_like_module_is_not_found(tmp_path, monkeypatch):
    """is_dir() False, even with an exact name match -> not found.
    Paired with the exact-match baseline: only is_dir()'s result
    differs, isolating that half of the outer and."""
    result, out = _run_convert(tmp_path, monkeypatch, {"tensor": "file"}, module="tensor")
    assert result == 1
    assert "Module not found: tensor" in out


def test_first_match_wins_and_stops_searching(tmp_path, monkeypatch):
    """The specific-module loop breaks on the first match; a module
    lacking a source file simply produces zero converted artifacts
    rather than falling through to a later, unrelated matching dir."""
    result, out = _run_convert(tmp_path, monkeypatch, {"01_tensor": "dir_no_src"}, module="01")
    assert result == 0
    assert "Converting 1 module" in out
    assert "Successfully generated 0 artifact" in out


# ---------------------------------------------------------------------------
# Ambiguous match resolution: which match wins must not depend on the order
# Path.iterdir() happens to return.
# ---------------------------------------------------------------------------


def _force_listing_order(src_dir, monkeypatch, *, reverse):
    """Pin `src_dir`'s iterdir() order, so an unsorted search fails the same
    way on every OS instead of passing on NTFS. Wraps _REAL_ITERDIR so
    patching twice in one test cannot stack reversals."""

    def patched_iterdir(self):
        entries = list(_REAL_ITERDIR(self))
        # resolve() both sides: tmp_path traverses the /tmp -> /private/tmp
        # symlink on macOS, so comparing unresolved paths never matches.
        if self.resolve() == src_dir.resolve():
            entries.sort(reverse=reverse)
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", patched_iterdir)


def test_ambiguous_suffix_match_resolves_to_lowest_module_number(tmp_path, monkeypatch):
    """Both dirs match module="attention" by the `_<module>` suffix. Under
    reverse-sorted listing only an explicit sorted() picks 12_attention.
    Fails without the fix on every OS."""
    _force_listing_order(tmp_path / "data" / "src", monkeypatch, reverse=True)

    result, out = _run_convert(
        tmp_path,
        monkeypatch,
        {"12_attention": "dir_with_src", "13_multi_head_attention": "dir_with_src"},
        module="attention",
    )

    assert result == 0
    assert "Converting 1 module" in out
    assert "12_attention" in out
    assert "13_multi_head_attention" not in out
    assert "Successfully generated 1 artifact" in out


def test_winner_is_identical_under_forward_and_reverse_listing_order(tmp_path, monkeypatch):
    """The winner must not depend on listing order at all: same query, two
    roots, forward then reverse listing, same module converted. Also defeats
    a 'fix' that hardcodes the name instead of sorting."""
    entries = {"13_multi_head_attention": "dir_with_src", "12_attention": "dir_with_src"}

    def converted_lines(root, reverse):
        _force_listing_order(root / "data" / "src", monkeypatch, reverse=reverse)
        result, out = _run_convert(root, monkeypatch, entries, module="attention")
        assert result == 0
        lines = [line for line in out.splitlines() if "→" in line]
        assert len(lines) == 1, f"expected exactly one converted module, got {lines!r}"
        return lines

    forward = converted_lines(tmp_path / "forward", reverse=False)
    reverse = converted_lines(tmp_path / "reverse", reverse=True)

    assert forward == reverse, f"winner depends on listing order: {forward!r} vs {reverse!r}"
    assert "12_attention" in forward[0]


def test_unambiguous_match_is_unaffected_by_listing_order(tmp_path, monkeypatch):
    """Sorting is a no-op for the single-candidate case that every existing
    exact/prefix/suffix test covers, none of which can tell whether the search
    is sorted. Guards against a fix that breaks the matchers."""
    _force_listing_order(tmp_path / "data" / "src", monkeypatch, reverse=True)

    result, out = _run_convert(tmp_path, monkeypatch, {"01_tensor": "dir_with_src"}, module="01")

    assert result == 0
    assert "Converting 1 module" in out
    assert "01_tensor" in out
    assert "Module not found" not in out


def test_all_branch_is_deterministic_under_reversed_listing_order(tmp_path, monkeypatch):
    """The "all" branch already sorted; pin that it stays sorted and agrees
    with the specific-module branch on ordering."""
    _force_listing_order(tmp_path / "data" / "src", monkeypatch, reverse=True)

    result, out = _run_convert(
        tmp_path,
        monkeypatch,
        {"20_capstone": "dir_with_src", "01_tensor": "dir_with_src"},
        module="all",
    )

    assert result == 0
    assert "Converting 2 module" in out
    converted = [line for line in out.splitlines() if "→" in line]
    assert len(converted) == 2, f"expected two converted modules, got {converted!r}"
    assert converted == sorted(converted), f"modules converted out of order: {converted!r}"


def test_convert_external_output_directory_succeeds(tmp_path, monkeypatch):
    """Verify that specifying an output directory outside project_root succeeds
    without raising ValueError on relative path calculation."""
    project_root = tmp_path / "project_root"
    external_out = tmp_path / "external_exports"
    src_dir = project_root / "data" / "src" / "01_tensor"
    src_dir.mkdir(parents=True)
    (src_dir / "01_tensor.py").write_text("x = 1\n", encoding="utf-8")

    import trentorch.export_sanitizer as sanitizer_module

    monkeypatch.setattr(sanitizer_module, "to_sandbox_code", lambda content: "code")

    cmd = ConvertCommand(CLIConfig.from_project_root(project_root))
    buf = StringIO()
    cmd.console = Console(file=buf, width=200, no_color=True)

    result = cmd.run(Namespace(module="01", format="py", out=str(external_out)))
    assert result == 0
    assert (external_out / "01_tensor.py").exists()
    assert (external_out / "01_tensor.py").read_text(encoding="utf-8") == "code"
    output = buf.getvalue()
    assert "01_tensor" in output
    assert "Successfully generated 1 artifact" in output

