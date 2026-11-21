"""Four sandboxed coding tools with strict path-jail and command allowlist."""

from __future__ import annotations

import difflib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from harness.models import ToolResult

IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
FORBIDDEN_SHELL_TOKENS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r")


class SandboxViolationError(Exception):
    """Raised when a tool attempts to access a path outside the workspace root."""


class WorkspaceSandbox:
    """Enforces path-jail containment inside a single workspace root directory."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self.root = Path(workspace_dir).resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Workspace directory does not exist: {self.root}")

    def resolve_path(self, raw_path: str) -> Path:
        if not isinstance(raw_path, str) or "\x00" in raw_path:
            raise SandboxViolationError("Invalid path: null byte or non-string input.")

        cleaned = raw_path.strip()
        if not cleaned:
            cleaned = "."

        candidate = Path(cleaned)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()

        if not resolved.is_relative_to(self.root):
            raise SandboxViolationError(
                f"Path escapes workspace jail: '{raw_path}' -> '{resolved}' (root: '{self.root}')"
            )
        return resolved

    def to_relative_posix(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()


class ToolRegistry:
    """Exposes the 4 core tools (T) and executes them inside WorkspaceSandbox."""

    def __init__(
        self,
        workspace_dir: Path | str,
        *,
        test_timeout_seconds: float = 30.0,
        max_read_lines: int = 400,
    ) -> None:
        self.sandbox = WorkspaceSandbox(workspace_dir)
        self.test_timeout_seconds = test_timeout_seconds
        self.max_read_lines = max_read_lines
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "edit_file": self.edit_file,
            "run_tests": self.run_tests,
        }

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._handlers.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._handlers

    def openai_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling tool schema definitions for the 4 core tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files inside a workspace-relative directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative directory path (default '.').",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file with 1-based line numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path."},
                            "start_line": {
                                "type": "integer",
                                "description": "1-based inclusive start line (default 1).",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "1-based inclusive end line (optional).",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace an exact unique substring in a workspace file and return unified diff.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path."},
                            "old_text": {"type": "string", "description": "Exact existing snippet to replace."},
                            "new_text": {"type": "string", "description": "Replacement text."},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run allowlisted test suite commands (pytest or unittest) in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Allowlisted test command, e.g. 'pytest -q' or 'python -m unittest'.",
                            }
                        },
                        "required": [],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_name=name,
                ok=False,
                error_code="UNKNOWN_TOOL",
                output=f"Tool '{name}' is not in allowed tool set {sorted(self._handlers.keys())}.",
            )
        try:
            return handler(**arguments)
        except SandboxViolationError as exc:
            return ToolResult(
                tool_name=name,
                ok=False,
                error_code="PATH_JAIL_VIOLATION",
                output=str(exc),
            )
        except TypeError as exc:
            return ToolResult(
                tool_name=name,
                ok=False,
                error_code="INVALID_ARGUMENTS",
                output=f"Invalid arguments for '{name}': {exc}",
            )

    def list_files(self, path: str = ".") -> ToolResult:
        target = self.sandbox.resolve_path(path)
        if not target.exists():
            return ToolResult(
                tool_name="list_files",
                ok=False,
                error_code="NOT_FOUND",
                output=f"Directory not found: {path}",
            )
        if not target.is_dir():
            return ToolResult(
                tool_name="list_files",
                ok=False,
                error_code="NOT_A_DIRECTORY",
                output=f"Path is not a directory: {path}",
            )

        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(
                d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".venv")
            )
            for fname in sorted(filenames):
                full_path = Path(dirpath) / fname
                try:
                    resolved = full_path.resolve()
                    if resolved.is_relative_to(self.sandbox.root):
                        files.append(self.sandbox.to_relative_posix(resolved))
                except OSError:
                    continue

        output = "\n".join(files) if files else "(empty directory)"
        return ToolResult(
            tool_name="list_files",
            ok=True,
            output=output,
            metadata={"count": len(files), "files": files},
        )

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> ToolResult:
        target = self.sandbox.resolve_path(path)
        if not target.exists() or not target.is_file():
            return ToolResult(
                tool_name="read_file",
                ok=False,
                error_code="FILE_NOT_FOUND",
                output=f"File not found: {path}",
            )

        if start_line < 1:
            return ToolResult(
                tool_name="read_file",
                ok=False,
                error_code="INVALID_LINE_RANGE",
                output=f"start_line must be >= 1, got {start_line}",
            )

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        effective_end = total if end_line is None else min(end_line, total)
        if end_line is not None and end_line < start_line:
            return ToolResult(
                tool_name="read_file",
                ok=False,
                error_code="INVALID_LINE_RANGE",
                output=f"end_line ({end_line}) < start_line ({start_line})",
            )

        if effective_end - start_line + 1 > self.max_read_lines:
            effective_end = start_line + self.max_read_lines - 1

        sliced = lines[start_line - 1 : effective_end]
        numbered = [f"L{idx}: {line}" for idx, line in enumerate(sliced, start=start_line)]
        rel = self.sandbox.to_relative_posix(target)
        header = f"# {rel} (lines {start_line}-{effective_end} of {total})"
        output = header + ("\n" + "\n".join(numbered) if numbered else "\n(empty)")
        return ToolResult(
            tool_name="read_file",
            ok=True,
            output=output,
            metadata={
                "path": rel,
                "start_line": start_line,
                "end_line": effective_end,
                "total_lines": total,
            },
        )

    def edit_file(self, path: str, old_text: str, new_text: str) -> ToolResult:
        target = self.sandbox.resolve_path(path)
        if not target.exists() or not target.is_file():
            return ToolResult(
                tool_name="edit_file",
                ok=False,
                error_code="FILE_NOT_FOUND",
                output=f"File not found: {path}",
            )
        if not old_text:
            return ToolResult(
                tool_name="edit_file",
                ok=False,
                error_code="EMPTY_OLD_TEXT",
                output="old_text must be non-empty.",
            )

        original = target.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences == 0:
            return ToolResult(
                tool_name="edit_file",
                ok=False,
                error_code="TARGET_NOT_FOUND",
                output=f"old_text not found in {path}.",
            )
        if occurrences > 1:
            return ToolResult(
                tool_name="edit_file",
                ok=False,
                error_code="AMBIGUOUS_TARGET",
                output=f"old_text matched {occurrences} locations in {path}; provide more context.",
            )

        updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")

        rel = self.sandbox.to_relative_posix(target)
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        diff_str = "".join(diff_lines)
        return ToolResult(
            tool_name="edit_file",
            ok=True,
            output=diff_str or f"Edited {rel} (no line diff).",
            diff=diff_str,
            metadata={"path": rel},
        )

    def _validate_test_command(self, command: str) -> list[str] | None:
        if not command or not command.strip():
            return [sys.executable, "-m", "pytest", "-q"]
        for bad in FORBIDDEN_SHELL_TOKENS:
            if bad in command:
                return None

        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return None
        if not tokens:
            return None

        # Reject path-traversal flags/targets escaping workspace
        for tok in tokens[1:]:
            if ".." in Path(tok).parts:
                return None

        head = tokens[0].lower()
        if head == "pytest":
            return [sys.executable, "-m", "pytest", *tokens[1:]]
        if head in {"python", "python3"} and len(tokens) >= 3 and tokens[1] == "-m":
            module = tokens[2]
            if module in {"pytest", "unittest"}:
                return [sys.executable, "-m", module, *tokens[3:]]
        return None

    def run_tests(self, command: str = "pytest -q") -> ToolResult:
        argv = self._validate_test_command(command)
        if argv is None:
            return ToolResult(
                tool_name="run_tests",
                ok=False,
                error_code="COMMAND_NOT_ALLOWED",
                output=(
                    f"Command rejected by allowlist: '{command}'. "
                    "Allowed commands: 'pytest ...', 'python -m pytest ...', 'python -m unittest ...'."
                ),
            )

        env = os.environ.copy()
        existing_py = env.get("PYTHONPATH", "")
        root_str = str(self.sandbox.root)
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_py}" if existing_py else root_str

        try:
            proc = subprocess.run(
                argv,
                cwd=self.sandbox.root,
                capture_output=True,
                text=True,
                timeout=self.test_timeout_seconds,
                shell=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                tool_name="run_tests",
                ok=False,
                error_code="TIMEOUT",
                output=f"Test command timed out after {self.test_timeout_seconds}s: {command}",
                metadata={"timeout_seconds": self.test_timeout_seconds, "stdout": exc.stdout or ""},
            )

        combined = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
        passed = proc.returncode == 0
        return ToolResult(
            tool_name="run_tests",
            ok=passed,
            error_code=None if passed else "TESTS_FAILED",
            output=combined or f"(exit code {proc.returncode})",
            metadata={"returncode": proc.returncode, "passed": passed, "command": command},
        )
