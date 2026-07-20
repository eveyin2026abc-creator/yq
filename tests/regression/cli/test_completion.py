from __future__ import annotations

import subprocess
from pathlib import Path

from cli.completion import (
    BASH_COMPLETION_COMMANDS,
    BLOCK_BEGIN,
    BLOCK_END,
    disable_tab_completion,
    enable_tab_completion,
    install_completion_files,
    render_bash_completion,
    render_powershell_completion,
)


def test_render_bash_completion_registers_python_and_msmodeling() -> None:
    script = render_bash_completion()

    assert "complete -o default -F _msmodeling_complete python python3 msmodeling" in script
    assert "cli.inference.text_generate" in script
    assert "cli.inference.model_adapter" in script
    assert "model_adapter_doctor" in script
    assert "--doctor-report" in script
    assert "--evidence-file" in script
    assert "--model-id" in script
    assert "--num-queries" in script
    assert "--quantize-linear-action" in script
    assert "--enable-tab-completion" in script
    assert "--disable-tab-completion" in script
    assert "--reload-shell" in script
    assert "--delete-completion-file" in script
    assert '"inference optix -tab ' not in script


def test_enable_tab_completion_writes_static_file_and_rc_block(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / "completion.bash"
    rc_file.write_text("# existing\n", encoding="utf-8")

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0

    contents = rc_file.read_text(encoding="utf-8")
    assert "# existing" in contents
    assert BLOCK_BEGIN in contents
    assert BLOCK_END in contents
    assert str(completion_file) in contents
    assert "complete -o default -F _msmodeling_complete" in completion_file.read_text(encoding="utf-8")


def test_enable_tab_completion_shell_quotes_rc_completion_path(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / 'completion-$(touch injected)-`date`-"quoted".bash'

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0

    contents = rc_file.read_text(encoding="utf-8")
    quoted_path = "'" + str(completion_file).replace("'", "'\"'\"'") + "'"
    assert f"if [ -r {quoted_path} ]; then" in contents
    assert f"    . {quoted_path}" in contents
    assert f'if [ -r "{completion_file}" ]; then' not in contents
    assert f'    . "{completion_file}"' not in contents


def test_enable_tab_completion_is_idempotent(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / "completion.bash"

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0
    first = rc_file.read_text(encoding="utf-8")

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0
    second = rc_file.read_text(encoding="utf-8")

    assert second == first
    assert second.count(BLOCK_BEGIN) == 1
    assert second.count(BLOCK_END) == 1


def test_enable_tab_completion_preserves_tail_when_managed_block_end_is_missing(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / "completion.bash"
    rc_file.write_text(
        "\n".join(
            (
                "# existing before",
                BLOCK_BEGIN,
                'if [ -r "/old/completion.bash" ]; then',
                "# user content that must survive",
                "export IMPORTANT=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0

    contents = rc_file.read_text(encoding="utf-8")
    assert "# user content that must survive" in contents
    assert "export IMPORTANT=1" in contents
    assert str(completion_file) in contents
    assert contents.count(BLOCK_BEGIN) == 2
    assert contents.count(BLOCK_END) == 1


def test_disable_tab_completion_removes_managed_block(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / "completion.bash"

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0
    assert disable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0

    contents = rc_file.read_text(encoding="utf-8")
    assert BLOCK_BEGIN not in contents
    assert BLOCK_END not in contents
    assert completion_file.exists()


def test_disable_tab_completion_preserves_tail_when_managed_block_end_is_missing(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / "completion.bash"
    rc_file.write_text(
        "\n".join(
            (
                "# existing before",
                BLOCK_BEGIN,
                'if [ -r "/old/completion.bash" ]; then',
                "# user content that must survive",
                "export IMPORTANT=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert disable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0

    contents = rc_file.read_text(encoding="utf-8")
    assert "# user content that must survive" in contents
    assert "export IMPORTANT=1" in contents
    assert BLOCK_BEGIN in contents


def test_disable_tab_completion_can_delete_static_file(tmp_path: Path) -> None:
    rc_file = tmp_path / ".bashrc"
    completion_file = tmp_path / "completion.bash"

    assert enable_tab_completion(shell="bash", rc_file=rc_file, completion_file=completion_file) == 0
    assert (
        disable_tab_completion(
            shell="bash",
            rc_file=rc_file,
            completion_file=completion_file,
            delete_completion_file=True,
        )
        == 0
    )

    assert not completion_file.exists()


def test_bash_completion_supports_model_adapter_subcommands() -> None:
    script = render_bash_completion()
    cases = (
        ("msmodeling inference model-adapter doctor --m", "--model-id"),
        ("msmodeling inference model-adapter verify --e", "--evidence-file"),
        ("msmodeling inference model-adapter export-evidence --d", "--doctor-report"),
        ("python -m cli.inference.model_adapter doctor --m", "--model-id"),
    )

    for command_line, expected in cases:
        words = command_line.split()
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                r"""
source "$1"
shift
COMP_WORDS=("$@")
COMP_CWORD=$((${#COMP_WORDS[@]} - 1))
_msmodeling_complete
printf '%s\n' "${COMPREPLY[@]}"
""",
                "bash",
                "/dev/stdin",
                *words,
            ],
            input=script,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert expected in result.stdout, f"{command_line} did not complete {expected}; got {result.stdout!r}"


def test_bash_completion_does_not_hijack_generic_python_modules() -> None:
    script = render_bash_completion()
    cases = (
        "python -m ",
        "python -m pip --",
        "python3 -m pytest --",
    )

    for command_line in cases:
        words = command_line.split()
        if command_line.endswith(" "):
            words.append("")
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                r"""
source "$1"
shift
COMP_WORDS=("$@")
COMP_CWORD=$((${#COMP_WORDS[@]} - 1))
_msmodeling_complete
printf '%s\n' "${COMPREPLY[@]}"
""",
                "bash",
                "/dev/stdin",
                *words,
            ],
            input=script,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == ""


def test_bash_completion_supports_explicit_python_module_prefix() -> None:
    script = render_bash_completion()
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            r"""
source "$1"
shift
COMP_WORDS=("$@")
COMP_CWORD=$((${#COMP_WORDS[@]} - 1))
_msmodeling_complete
printf '%s\n' "${COMPREPLY[@]}"
""",
            "bash",
            "/dev/stdin",
            "python",
            "-m",
            "cli.inference.",
        ],
        input=script,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "cli.inference.text_generate" in result.stdout
    assert "cli.inference.model_adapter" in result.stdout


def test_render_powershell_completion_registers_python_and_msmodeling() -> None:
    script = render_powershell_completion()

    assert 'Register-ArgumentCompleter -Native -CommandName @("python", "python3", "msmodeling")' in script
    assert 'if ($commandName -eq "msmodeling")' in script
    assert (
        '@("inference", "optix", "--enable-tab-completion", "--disable-tab-completion", '
        '"--reload-shell", "--delete-completion-file", "--help")'
    ) in script
    assert '@("text-generate", "throughput-optimizer", "model-adapter", "video-generate", "--help")' in script
    assert '@("doctor", "verify", "export-evidence", "--help")' in script
    assert "'model_adapter_doctor'" in script
    assert "'model_adapter_verify'" in script
    assert "'model_adapter_export_evidence'" in script
    assert "--doctor-report" in script
    assert "--evidence-file" in script
    assert "--model-id" in script
    assert 'if ($words.Count -gt 1 -and $words[$words.Count - 2] -eq "-m")' not in script


def test_install_completion_files_writes_bash_completion_command_names(tmp_path: Path) -> None:
    assert install_completion_files(prefix=str(tmp_path)) == 0

    completions_dir = tmp_path / "share" / "bash-completion" / "completions"

    for command in BASH_COMPLETION_COMMANDS:
        completion_file = completions_dir / command

        assert completion_file.exists()
        assert "complete -o default -F _msmodeling_complete python python3 msmodeling" in completion_file.read_text(
            encoding="utf-8"
        )

        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                r"""
source "$1"
complete -p "$2"
if [[ "$2" == "msmodeling" ]]; then
    COMP_WORDS=("$2" inference text-generate --n)
else
    COMP_WORDS=("$2" -m cli.inference.text_generate --n)
fi
COMP_CWORD=3
_msmodeling_complete
printf '%s\n' "${COMPREPLY[@]}"
""",
                "bash",
                str(completion_file),
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert f"complete -o default -F _msmodeling_complete {command}" in result.stdout
        assert "--num-queries" in result.stdout
