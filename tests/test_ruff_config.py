"""Ruff 配置的回归测试。

这些测试锁定的不是「代码风格」，而是「配置本身」：
一旦 pyproject.toml 中的 Ruff 配置被误删或改回默认值，
规则集会随 Ruff 版本漂移，CI 与本地行为也会不一致。
"""

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
GITIGNORE = PROJECT_ROOT / ".gitignore"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def _find_ruff() -> str:
    """优先使用虚拟环境中的 ruff，其次是 PATH 中的 ruff。"""
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "ruff.exe",  # Windows
        PROJECT_ROOT / ".venv" / "bin" / "ruff",  # POSIX
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    found = shutil.which("ruff")
    if found:
        return found

    pytest.skip("未找到 ruff 可执行文件")
    raise AssertionError("unreachable")  # pragma: no cover


def _run_ruff(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_find_ruff(), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def _python_files() -> list[str]:
    """显式列出待检查的 Python 文件。

    刻意不使用 `.` 让 Ruff 自行遍历：在受限环境中遍历目录会触发
    `ruff format` 的 panic（Ruff 0.16.7 的已知脆弱点），显式文件列表
    可让测试在任何环境下得到一致结果。
    """
    files: list[str] = []
    for directory in ("src", "tests", "examples", "docs"):
        root = PROJECT_ROOT / directory
        if root.is_dir():
            files.extend(
                path.relative_to(PROJECT_ROOT).as_posix() for path in sorted(root.rglob("*.py"))
            )
    assert files, "未找到任何 Python 文件"
    return files


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ruff_settings() -> str:
    """`ruff check --show-settings` 的实际生效输出（而非仅看配置文件文本）。"""
    result = _run_ruff("check", "--show-settings", "--no-cache", ".")
    assert result.returncode == 0, f"ruff --show-settings 执行失败:\n{result.stderr}"
    return result.stdout


def _setting(settings: str, key: str) -> str | None:
    match = re.search(rf"^\s*(?:\w+\.)*{re.escape(key)} = (.+)$", settings, re.MULTILINE)
    return match.group(1).strip().strip('"') if match else None


class TestRuffConfig:
    def test_pyproject_contains_ruff_section(self, pyproject: dict) -> None:
        assert "ruff" in pyproject.get("tool", {}), "pyproject.toml 缺少 [tool.ruff] 配置段"

    def test_target_version_is_pinned(self, pyproject: dict, ruff_settings: str) -> None:
        """目标版本必须写死，否则会随解释器环境漂移。"""
        configured = pyproject["tool"]["ruff"]["target-version"]
        assert configured == "py313"
        # Ruff 在 --show-settings 中把 py313 归一化显示为 3.13
        assert _setting(ruff_settings, "target_version") == "3.13"

    def test_line_length_is_explicit(self, pyproject: dict, ruff_settings: str) -> None:
        assert pyproject["tool"]["ruff"]["line-length"] == 100
        assert _setting(ruff_settings, "line_length") == "100"

    def test_rule_set_is_pinned(self, pyproject: dict) -> None:
        """显式 select 固定规则集，避免升级 Ruff 后突然出现新告警。"""
        select = pyproject["tool"]["ruff"]["lint"]["select"]
        assert len(select) >= 10, f"select 过于宽松，可能未固定规则集: {select}"
        assert {"E", "F", "I", "UP", "B", "RUF"} <= set(select)

    def test_ambiguous_unicode_rules_disabled(self, pyproject: dict) -> None:
        """中文注释中的全角标点是刻意为之，不应被判定为易混字符。"""
        ignore = pyproject["tool"]["ruff"]["lint"]["ignore"]
        assert {"RUF001", "RUF002", "RUF003"} <= set(ignore)

    def test_first_party_imports_declared(self, pyproject: dict) -> None:
        assert pyproject["tool"]["ruff"]["lint"]["isort"]["known-first-party"] == ["aier"]

    def test_ruff_is_a_dev_dependency_not_runtime(self, pyproject: dict) -> None:
        """Ruff 是开发工具，不应出现在运行时依赖中。"""
        runtime = " ".join(pyproject["project"]["dependencies"])
        assert "ruff" not in runtime, "ruff 不应作为运行时依赖"

        dev = " ".join(pyproject["dependency-groups"]["dev"])
        assert "ruff" in dev, "ruff 应位于 [dependency-groups] dev 中"

    def test_cache_dir_is_gitignored(self) -> None:
        lines = {line.strip() for line in GITIGNORE.read_text(encoding="utf-8").splitlines()}
        assert ".ruff_cache" in lines or ".ruff_cache/" in lines, ".ruff_cache 未加入 .gitignore"

    def test_ci_workflow_exists(self) -> None:
        assert CI_WORKFLOW.is_file(), "缺少 CI 工作流"
        content = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "ruff check" in content
        assert "ruff format --check" in content


class TestRuffRunsClean:
    def test_lint_is_clean(self) -> None:
        result = _run_ruff("check", *_python_files(), "--no-cache", "--output-format", "concise")
        assert result.returncode == 0, f"ruff check 未通过:\n{result.stdout}"
        assert "Found" not in result.stdout, result.stdout

    def test_formatting_is_stable(self) -> None:
        result = _run_ruff("format", "--check", "--no-cache", *_python_files())
        assert result.returncode == 0, (
            f"ruff format --check 未通过:\n{result.stdout}{result.stderr}"
        )


class TestOpenAIModelAnnotations:
    """修复 Optional 未定义的崩溃缺陷后留下的回归测试。

    此前 openai.py 顶部移除了 `Optional` 导入，但函数体内仍在注解中使用它。
    由于是延迟导入，F821 在运行期才会表现为 NameError。
    """

    def test_module_source_has_no_optional_reference(self) -> None:
        source = (PROJECT_ROOT / "src" / "aier" / "ai" / "providers" / "openai.py").read_text(
            encoding="utf-8"
        )
        assert "Optional" not in source, "openai.py 中仍残留 Optional 引用"

    def test_stream_invoke_annotations_resolve(self) -> None:
        """执行全部注解求值，确保不再有未定义名称。"""
        from aier.ai.providers import openai as openai_provider

        for member in vars(openai_provider).values():
            if getattr(member, "__module__", None) != openai_provider.__name__:
                continue
            for annotation in getattr(member, "__annotations__", {}).values():
                # 字符串形式的注解若含未定义名称，这里会抛 NameError
                if isinstance(annotation, str):
                    eval(annotation, vars(openai_provider))


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 13), "项目要求 Python >= 3.13"
