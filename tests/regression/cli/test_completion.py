from __future__ import annotations

from pathlib import Path

from cli.completion import (
    BLOCK_BEGIN,
    BLOCK_END,
    enable_tab_completion,
    render_bash_completion,
)


def test_render_bash_completion_registers_python_and_msmodeling() -> None:
    script = render_bash_completion()

    assert "complete -o default -F _msmodeling_complete python python3 msmodeling" in script
    assert "cli.inference.text_generate" in script
    assert "--num-queries" in script
    assert "--quantize-linear-action" in script
    assert "--enable-tab-completion" in script
    assert "-tab" in script


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

