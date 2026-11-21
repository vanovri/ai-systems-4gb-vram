"""Unit tests for the 4 sandboxed tools, path-jail enforcement, and command allowlist."""

from __future__ import annotations

from pathlib import Path

from harness.tools import ToolRegistry


def test_list_files_lists_workspace_and_ignores_cache(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "helper.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"pyc")

    tools = ToolRegistry(tmp_path)
    res = tools.execute("list_files", {"path": "."})
    assert res.ok is True
    assert res.metadata["files"] == ["app.py", "sub/helper.py"]


def test_path_jail_blocks_parent_traversal_and_absolute_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("top-secret", encoding="utf-8")

    tools = ToolRegistry(workspace)

    # Relative ../ traversal
    res_rel = tools.execute("read_file", {"path": "../secret.txt"})
    assert res_rel.ok is False
    assert res_rel.error_code == "PATH_JAIL_VIOLATION"

    # Absolute path outside workspace
    res_abs = tools.execute("read_file", {"path": str(outside)})
    assert res_abs.ok is False
    assert res_abs.error_code == "PATH_JAIL_VIOLATION"

    # Traversal on list_files and edit_file
    assert tools.execute("list_files", {"path": ".."}).error_code == "PATH_JAIL_VIOLATION"
    assert (
        tools.execute("edit_file", {"path": "../secret.txt", "old_text": "top", "new_text": "x"}).error_code
        == "PATH_JAIL_VIOLATION"
    )


def test_read_file_line_slicing(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    res = tools.execute("read_file", {"path": "sample.py", "start_line": 2, "end_line": 3})
    assert res.ok is True
    assert "L2: line2" in res.output
    assert "L3: line3" in res.output
    assert "L1:" not in res.output
    assert res.metadata["total_lines"] == 4


def test_edit_file_produces_unified_diff_and_rejects_ambiguous(tmp_path: Path) -> None:
    target = tmp_path / "math_ops.py"
    target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    res = tools.execute(
        "edit_file",
        {"path": "math_ops.py", "old_text": "return a - b", "new_text": "return a + b"},
    )
    assert res.ok is True
    assert res.diff is not None
    assert "-    return a - b" in res.diff
    assert "+    return a + b" in res.diff
    assert "return a + b" in target.read_text(encoding="utf-8")

    # Missing target
    miss = tools.execute(
        "edit_file",
        {"path": "math_ops.py", "old_text": "nonexistent", "new_text": "x"},
    )
    assert miss.ok is False
    assert miss.error_code == "TARGET_NOT_FOUND"

    # Ambiguous target
    dup = tmp_path / "dup.py"
    dup.write_text("x = 1\nx = 1\n", encoding="utf-8")
    ambig = tools.execute("edit_file", {"path": "dup.py", "old_text": "x = 1", "new_text": "x = 2"})
    assert ambig.ok is False
    assert ambig.error_code == "AMBIGUOUS_TARGET"


def test_run_tests_allowlist_and_shell_injection_block(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_one():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    ok_res = tools.execute("run_tests", {"command": "pytest -q"})
    assert ok_res.ok is True
    assert ok_res.metadata["returncode"] == 0

    for dangerous in [
        "rm -rf /",
        "pytest -q && echo pwned",
        "pytest -q; ls",
        "python -c 'print(1)'",
        "powershell -Command dir",
    ]:
        bad = tools.execute("run_tests", {"command": dangerous})
        assert bad.ok is False
        assert bad.error_code == "COMMAND_NOT_ALLOWED"


def test_run_tests_enforces_timeout(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text(
        "import time\ndef test_sleep():\n    time.sleep(3)\n",
        encoding="utf-8",
    )
    tools = ToolRegistry(tmp_path, test_timeout_seconds=0.5)
    res = tools.execute("run_tests", {"command": "pytest -q"})
    assert res.ok is False
    assert res.error_code == "TIMEOUT"
