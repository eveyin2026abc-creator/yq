from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from cli.completion import (
    BASH_COMPLETION_COMMANDS,
    BLOCK_BEGIN,
    BLOCK_END,
    _bash_rc_file,
    _completion_for_shell,
    _default_prefix,
    _detect_shell,
    build_parser,
    disable_tab_completion,
    enable_tab_completion,
    install_completion_files,
    main as completion_main,
    print_completion,
    render_bash_completion,
    render_powershell_completion,
)
from tests.helpers.cli_runner import run_cli_main


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
    assert "-tab" in script
    assert "--enable-tab-completion" in script
    assert "--disable-tab-completion" in script
    assert "--reload-shell" in script
    assert "--delete-completion-file" in script


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


def test_detect_shell_prefers_parent_process_comm(monkeypatch) -> None:
    monkeypatch.setattr("os.getppid", lambda: 1234)
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: "pwsh\n")

    assert _detect_shell() == "powershell"


def test_detect_shell_falls_back_to_environment(monkeypatch) -> None:
    def raise_os_error(self: Path, encoding: str | None = None) -> str:
        raise OSError

    monkeypatch.setattr(Path, "read_text", raise_os_error)
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")

    assert _detect_shell() == "zsh"


def test_default_prefix_uses_userbase_or_sys_prefix(monkeypatch, tmp_path: Path) -> None:
    userbase = tmp_path / "userbase"
    system_prefix = tmp_path / "system"
    monkeypatch.setattr("site.getuserbase", lambda: str(userbase))
    monkeypatch.setattr("sys.prefix", str(system_prefix))

    assert _default_prefix(user=True) == userbase
    assert _default_prefix(user=False) == system_prefix


def test_bash_rc_file_uses_home_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _bash_rc_file() == tmp_path / ".bashrc"


def test_completion_for_shell_selects_renderer() -> None:
    assert "bashcompinit" in _completion_for_shell("zsh")
    assert "Register-ArgumentCompleter" in _completion_for_shell("powershell")
    assert "complete -o default -F _msmodeling_complete" in _completion_for_shell("bash")


def test_print_completion_writes_resolved_completion(capsys) -> None:
    assert print_completion("bash") == 0

    assert "complete -o default -F _msmodeling_complete" in capsys.readouterr().out


def test_build_parser_accepts_completion_subcommands() -> None:
    parser = build_parser()

    install_args = parser.parse_args(["--shell", "zsh", "install", "--prefix", "/tmp/msmodeling", "--user"])
    print_args = parser.parse_args(["print", "--shell", "powershell"])

    assert install_args.command == "install"
    assert install_args.shell == "zsh"
    assert install_args.prefix == "/tmp/msmodeling"
    assert install_args.user is True
    assert print_args.command == "print"
    assert print_args.print_shell == "powershell"


def test_completion_main_dispatches_install() -> None:
    with patch("cli.completion.install_completion_files", return_value=7) as install:
        result = run_cli_main(
            completion_main, ["install", "--prefix", "/tmp/msmodeling", "--user"], prog="msmodeling-tab"
        )

    assert result.returncode == 7
    install.assert_called_once_with("/tmp/msmodeling", True)


def test_completion_main_dispatches_enable() -> None:
    with patch("cli.completion.enable_tab_completion", return_value=3) as enable:
        result = run_cli_main(completion_main, ["--shell", "bash", "enable"], prog="msmodeling-tab")

    assert result.returncode == 3
    enable.assert_called_once_with("bash")


def test_completion_main_defaults_to_print() -> None:
    with patch("cli.completion.print_completion", return_value=5) as printer:
        result = run_cli_main(completion_main, ["--shell", "zsh"], prog="msmodeling-tab")

    assert result.returncode == 5
    printer.assert_called_once_with("zsh")


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
