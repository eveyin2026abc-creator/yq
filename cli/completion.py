"""Static shell completion for msmodeling commands.

The generated completion is pure shell code. Pressing Tab does not start Python
or import heavyweight runtime dependencies.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import site
import sys


MODULE_TARGETS = {
    "cli.inference.text_generate": "text_generate",
    "cli.inference.video_generate": "video_generate",
    "cli.inference.throughput_optimizer": "throughput_optimizer",
    "serving_cast.main": "serving_cast_main",
}

MODULE_SUBCOMMAND_TARGETS = {
    "cli.inference.model_adapter": {
        "doctor": "model_adapter_doctor",
        "verify": "model_adapter_verify",
        "export-evidence": "model_adapter_export_evidence",
    },
}

CLI_TARGETS = {
    "text-generate": "text_generate",
    "video-generate": "video_generate",
    "throughput-optimizer": "throughput_optimizer",
}

CLI_SUBCOMMAND_TARGETS = {
    "model-adapter": MODULE_SUBCOMMAND_TARGETS["cli.inference.model_adapter"],
}

MODULES = (*MODULE_TARGETS.keys(), *MODULE_SUBCOMMAND_TARGETS.keys())

TOP_LEVEL_OPTIONS = (
    "--enable-tab-completion",
    "--disable-tab-completion",
    "--reload-shell",
    "--delete-completion-file",
    "--help",
)

COMMON_OPTIONS = (
    "--device",
    "--num-devices",
    "--reserved-memory-gb",
    "--log-level",
)

OPTIONS = {
    "text_generate": (
        *COMMON_OPTIONS,
        "--num-queries",
        "--query-length",
        "--context-length",
        "--decode",
        "--prefix-cache-hit-rate",
        "--num-mtp-tokens",
        "--disable-repetition",
        "--compile",
        "--compile-allow-graph-break",
        "--enable-sequence-parallel",
        "--quantize-linear-action",
        "--quantize-non-expert-linear-action",
        "--quantize-lmhead",
        "--mxfp4-group-size",
        "--quantize-attention-action",
        "--graph-log-url",
        "--dump-input-shapes",
        "--dump-op-bound-results",
        "--chrome-trace",
        "--num-hidden-layers-override",
        "--tp-size",
        "--pp-size",
        "--dp-size",
        "--ep-size",
        "--o-proj-tp-size",
        "--o-proj-dp-size",
        "--mlp-tp-size",
        "--mlp-dp-size",
        "--lmhead-tp-size",
        "--lmhead-dp-size",
        "--moe-tp-size",
        "--moe-dp-size",
        "--word-embedding-tp",
        "--enable-redundant-experts",
        "--enable-shared-expert-tp",
        "--enable-external-shared-experts",
        "--host-external-shared-experts",
        "--vision-tp-size",
        "--image-batch-size",
        "--image-height",
        "--image-width",
        "--remote-source",
        "--performance-model",
        "--profiling-database",
        "--disable-profiling-interpolation",
        "--export-empirical-metrics",
        "--help",
    ),
    "video_generate": (
        "--device",
        "--batch-size",
        "--seq-len",
        "--chrome-trace",
        "--height",
        "--width",
        "--frame-num",
        "--sample-step",
        "--log-level",
        "--dtype",
        "--remote-source",
        "--quantize-linear-action",
        "--quantize-attention-action",
        "--use-cfg",
        "--world-size",
        "--ulysses-size",
        "--cfg-parallel",
        "--dit-cache",
        "--cache-step-range",
        "--cache-step-interval",
        "--cache-block-range",
        "--help",
    ),
    "throughput_optimizer": (
        *COMMON_OPTIONS,
        "--input-length",
        "--output-length",
        "--compile",
        "--compile-allow-graph-break",
        "--num-mtp-tokens",
        "--mtp-acceptance-rate",
        "--prefix-cache-hit-rate",
        "--quantize-linear-action",
        "--quantize-non-expert-linear-action",
        "--mxfp4-group-size",
        "--quantize-attention-action",
        "--tp-sizes",
        "--ep-sizes",
        "--moe-dp-sizes",
        "--enable-shared-expert-tp",
        "--enable-sequence-parallel",
        "--enable-dispatch-ffn-combine",
        "--word-embedding-tp",
        "--performance-model",
        "--profiling-database",
        "--chrome-trace",
        "--ttft-limits",
        "--tpot-limits",
        "--max-batched-tokens",
        "--batch-range",
        "--serving-cost",
        "--disagg",
        "--jobs",
        "--max-search-combinations",
        "--concurrency-search-strategy",
        "--dump-original-results",
        "--image-batch-size",
        "--image-height",
        "--image-width",
        "--prefill-devices-per-instance",
        "--decode-devices-per-instance",
        "--enable-optimize-prefill-decode-ratio",
        "--help",
    ),
    "serving_cast_main": (
        "--instance_config_path",
        "--common_config_path",
        "--enable_profiling",
        "--profiling_output_path",
        "--output_json",
        "--help",
    ),
    "model_adapter_doctor": (
        *COMMON_OPTIONS,
        "--model-id",
        "--model_id",
        "--num-queries",
        "--query-length",
        "--context-length",
        "--decode",
        "--compile",
        "--compile-allow-graph-break",
        "--dump-input-shapes",
        "--num-hidden-layers-override",
        "--remote-source",
        "--disable-repetition",
        "--quantize-linear-action",
        "--quantize-attention-action",
        "--image-batch-size",
        "--image-height",
        "--image-width",
        "--tp-size",
        "--dp-size",
        "--ep-size",
        "--moe-tp-size",
        "--moe-dp-size",
        "--vision-tp-size",
        "--from-command-file",
        "--raw-insight-file",
        "--hints-file",
        "--patch-failure-file",
        "--ignore-existing-profile",
        "--profile-draft-output",
        "--output",
        "--help",
    ),
    "model_adapter_verify": (
        *COMMON_OPTIONS,
        "--model-id",
        "--model_id",
        "--evidence-file",
        "--output",
        "--st-case-output",
        "--num-queries",
        "--query-length",
        "--context-length",
        "--decode",
        "--num-hidden-layers-override",
        "--disable-repetition",
        "--performance-model",
        "--profiling-database",
        "--tp-size",
        "--dp-size",
        "--ep-size",
        "--moe-tp-size",
        "--moe-dp-size",
        "--vision-tp-size",
        "--remote-source",
        "--help",
    ),
    "model_adapter_export_evidence": (
        "--doctor-report",
        "--output",
        "--help",
    ),
}

BLOCK_BEGIN = "# >>> msmodeling completion >>>"
BLOCK_END = "# <<< msmodeling completion <<<"
DATA_DIR = Path.home() / ".local" / "share" / "msmodeling"
BASH_COMPLETION_FILE = DATA_DIR / "completion.bash"
BASH_COMPLETION_COMMANDS = ("python", "python3", "msmodeling")


def _words(values: tuple[str, ...]) -> str:
    return " ".join(values)


def _ps_array(values: tuple[str, ...]) -> str:
    return "@(" + ", ".join(f"'{value}'" for value in values) + ")"


def _module_script_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _bash_path_patterns(module: str) -> tuple[str, ...]:
    path = _module_script_path(module)
    return (f"*/{path}", path, Path(path).name)


def _ps_path_patterns(module: str) -> tuple[str, ...]:
    path = _module_script_path(module)
    windows_path = path.replace("/", "\\")
    return (f"*/{path}", f"*\\{windows_path}", path, windows_path, Path(path).name)


def _bash_subcommand_case(subtargets: dict[str, str], word_expr: str) -> str:
    lines = [f'                    case "{word_expr}" in']
    lines.extend(f"                        {command}) echo {target}; return 0 ;;" for command, target in subtargets.items())
    lines.append("                    esac")
    return "\n".join(lines)


def _bash_complete_options_cases() -> str:
    return "\n".join(
        (
            f"        {target})\n"
            f'            COMPREPLY=( $(compgen -W "{_words(options)}" -- "$cur") )\n'
            "            ;;"
        )
        for target, options in OPTIONS.items()
    )


def _bash_cli_target_cases() -> str:
    lines = [f"        {command}) echo {target}; return 0 ;;" for command, target in CLI_TARGETS.items()]
    for command, subtargets in CLI_SUBCOMMAND_TARGETS.items():
        lines.append(f"        {command})")
        lines.append(_bash_subcommand_case(subtargets, "${COMP_WORDS[3]}"))
        lines.append("            ;;")
    return "\n".join(lines)


def _bash_module_target_cases() -> str:
    lines: list[str] = []
    for module, target in MODULE_TARGETS.items():
        lines.append(f"                {module}) echo {target}; return 0 ;;")
    for module, subtargets in MODULE_SUBCOMMAND_TARGETS.items():
        lines.append(f"                {module})")
        lines.append(_bash_subcommand_case(subtargets, "${COMP_WORDS[i + 2]}"))
        lines.append("                    ;;")
    return "\n".join(lines)


def _bash_path_target_cases() -> str:
    lines: list[str] = []
    for module, target in MODULE_TARGETS.items():
        lines.append(f"            {'|'.join(_bash_path_patterns(module))})")
        lines.append(f"                echo {target}; return 0 ;;")
    for module, subtargets in MODULE_SUBCOMMAND_TARGETS.items():
        lines.append(f"            {'|'.join(_bash_path_patterns(module))})")
        lines.append(_bash_subcommand_case(subtargets, "$next"))
        lines.append("                ;;")
    return "\n".join(lines)


def _ps_subcommand_switch(subtargets: dict[str, str], indent: str = "                ") -> str:
    return "\n".join(f'{indent}"{command}" {{ return "{target}" }}' for command, target in subtargets.items())


def _ps_options_entries() -> str:
    return "\n".join(f"    '{target}' = {_ps_array(options)}" for target, options in OPTIONS.items())


def _ps_cli_target_switch() -> str:
    lines = [f'        "{command}" {{ return "{target}" }}' for command, target in CLI_TARGETS.items()]
    for command, subtargets in CLI_SUBCOMMAND_TARGETS.items():
        lines.append(f'        "{command}" {{')
        lines.append("            switch ($Words[3]) {")
        lines.append(_ps_subcommand_switch(subtargets))
        lines.append("            }")
        lines.append("        }")
    return "\n".join(lines)


def _ps_module_target_switch() -> str:
    lines: list[str] = []
    for module, target in MODULE_TARGETS.items():
        lines.append(f'                "{module}" {{ return "{target}" }}')
    for module, subtargets in MODULE_SUBCOMMAND_TARGETS.items():
        lines.append(f'                "{module}" {{')
        lines.append("                    switch ($Words[$i + 2]) {")
        lines.append(_ps_subcommand_switch(subtargets, indent="                        "))
        lines.append("                    }")
        lines.append("                }")
    return "\n".join(lines)


def _ps_path_target_switch() -> str:
    lines: list[str] = []
    for module, target in MODULE_TARGETS.items():
        lines.extend(f'            "{pattern}" {{ return "{target}" }}' for pattern in _ps_path_patterns(module))
    for module, subtargets in MODULE_SUBCOMMAND_TARGETS.items():
        for pattern in _ps_path_patterns(module):
            lines.append(f'            "{pattern}" {{')
            lines.append("                switch ($next) {")
            lines.append(_ps_subcommand_switch(subtargets, indent="                    "))
            lines.append("                }")
            lines.append("            }")
    return "\n".join(lines)


def render_bash_completion() -> str:
    """Return a self-contained bash completion script."""
    return f"""# Static completion for msmodeling.
# Generated by msmodeling --enable-tab-completion. Re-run after CLI option changes.

_msmodeling_complete()
{{
    local cmd cur prev target
    COMPREPLY=()
    cmd="${{COMP_WORDS[0]}}"
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev=""
    if [[ $COMP_CWORD -gt 0 ]]; then
        prev="${{COMP_WORDS[COMP_CWORD - 1]}}"
    fi

    if [[ "$cmd" == "msmodeling" ]]; then
        _msmodeling_complete_cli_command
        return 0
    fi

    case "$cur" in
        cli.inference.*)
            COMPREPLY=( $(compgen -W "{_words(MODULES)}" -- "$cur") )
            return 0
            ;;
    esac

    target="$(_msmodeling_completion_target)"
    if [[ -z "$target" || "$cur" != --* ]]; then
        return 0
    fi

    _msmodeling_complete_options "$target" "$cur"
}}

_msmodeling_complete_cli_command()
{{
    local cur first second target
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    first="${{COMP_WORDS[1]}}"
    second="${{COMP_WORDS[2]}}"

    if [[ $COMP_CWORD -eq 1 ]]; then
        if [[ "$cur" == --* ]]; then
            COMPREPLY=( $(compgen -W "{_words(TOP_LEVEL_OPTIONS)}" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "inference optix {_words(TOP_LEVEL_OPTIONS)}" -- "$cur") )
        fi
        return 0
    fi

    if [[ "$first" == "inference" && $COMP_CWORD -eq 2 ]]; then
        COMPREPLY=( $(compgen -W "text-generate throughput-optimizer model-adapter video-generate --help" -- "$cur") )
        return 0
    fi

    if [[ "$first" == "inference" && "$second" == "model-adapter" && $COMP_CWORD -eq 3 ]]; then
        COMPREPLY=( $(compgen -W "doctor verify export-evidence --help" -- "$cur") )
        return 0
    fi

    target="$(_msmodeling_cli_completion_target)"
    if [[ -z "$target" || "$cur" != --* ]]; then
        return 0
    fi
    _msmodeling_complete_options "$target" "$cur"
}}

_msmodeling_complete_options()
{{
    local target cur
    target="$1"
    cur="$2"

    case "$target" in
{_bash_complete_options_cases()}
    esac
}}

_msmodeling_cli_completion_target()
{{
    case "${{COMP_WORDS[2]}}" in
{_bash_cli_target_cases()}
    esac
}}

_msmodeling_completion_target()
{{
    local i word next
    for ((i = 1; i < COMP_CWORD; i++)); do
        word="${{COMP_WORDS[i]}}"
        next="${{COMP_WORDS[i + 1]}}"

        if [[ "$word" == "-m" ]]; then
            case "$next" in
{_bash_module_target_cases()}
            esac
        fi

        case "$word" in
{_bash_path_target_cases()}
        esac
    done
}}

complete -o default -F _msmodeling_complete python python3 msmodeling
"""


def render_zsh_completion() -> str:
    """Return a self-contained zsh completion script via bashcompinit."""
    return (
        "# Static completion for msmodeling in zsh via bashcompinit.\n"
        "autoload -Uz +X compinit && compinit\n"
        "autoload -Uz +X bashcompinit && bashcompinit\n"
        + render_bash_completion()
    )


def render_powershell_completion() -> str:
    """Return a self-contained PowerShell completion script."""
    return f"""# Static completion for msmodeling.
# Generated by msmodeling --enable-tab-completion. Re-run after CLI option changes.

$script:MsmodelingModules = {_ps_array(MODULES)}
$script:MsmodelingOptions = @{{
{_ps_options_entries()}
}}

function Get-MsmodelingCliCompletionTarget {{
    param([string[]] $Words)

    switch ($Words[2]) {{
{_ps_cli_target_switch()}
    }}
}}

function Get-MsmodelingCompletionTarget {{
    param([string[]] $Words)

    for ($i = 1; $i -lt $Words.Count; $i++) {{
        $word = $Words[$i]
        $next = if ($i + 1 -lt $Words.Count) {{ $Words[$i + 1] }} else {{ "" }}

        if ($word -eq "-m") {{
            switch ($next) {{
{_ps_module_target_switch()}
            }}
        }}

        switch -Wildcard ($word) {{
{_ps_path_target_switch()}
        }}
    }}
}}

Register-ArgumentCompleter -Native -CommandName @("python", "python3", "msmodeling") -ScriptBlock {{
    param($wordToComplete, $commandAst, $cursorPosition)

    $words = @($commandAst.CommandElements | ForEach-Object {{ $_.ToString() }})
    $commandName = if ($words.Count -gt 0) {{ $words[0] }} else {{ "" }}

    if ($commandName -eq "msmodeling") {{
        if ($words.Count -gt 1 -and $words[1] -eq "inference" -and $words.Count -eq 2) {{
            @("text-generate", "throughput-optimizer", "model-adapter", "video-generate", "--help") |
                Where-Object {{ $_ -like "$wordToComplete*" }} |
                ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterValue", $_) }}
            return
        }}

        if ($words.Count -le 2) {{
            @("inference", "optix", "--enable-tab-completion", "--disable-tab-completion", "--reload-shell", "--delete-completion-file", "--help") |
                Where-Object {{ $_ -like "$wordToComplete*" }} |
                ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterValue", $_) }}
            return
        }}

        if ($words.Count -gt 2 -and $words[1] -eq "inference" -and $words[2] -eq "model-adapter" -and $words.Count -eq 3) {{
            @("doctor", "verify", "export-evidence", "--help") |
                Where-Object {{ $_ -like "$wordToComplete*" }} |
                ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterValue", $_) }}
            return
        }}

        $target = Get-MsmodelingCliCompletionTarget -Words $words
        if (-not $target -or -not $wordToComplete.StartsWith("--")) {{
            return
        }}

        $script:MsmodelingOptions[$target] |
            Where-Object {{ $_ -like "$wordToComplete*" }} |
            ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterName", $_) }}
        return
    }}

    if ($wordToComplete -like "cli.inference.*") {{
        $script:MsmodelingModules |
            Where-Object {{ $_ -like "$wordToComplete*" }} |
            ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterValue", $_) }}
        return
    }}

    $target = Get-MsmodelingCompletionTarget -Words $words
    if (-not $target -or -not $wordToComplete.StartsWith("--")) {{
        return
    }}

    $script:MsmodelingOptions[$target] |
        Where-Object {{ $_ -like "$wordToComplete*" }} |
        ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterName", $_) }}
}}
"""


def _detect_shell() -> str:
    try:
        comm = Path(f"/proc/{os.getppid()}/comm").read_text(encoding="utf-8").strip()
        if comm in {"bash", "zsh"}:
            return comm
        if comm in {"pwsh", "powershell"}:
            return "powershell"
    except OSError:
        pass

    shell_name = Path(os.environ.get("SHELL", "")).name.lower()
    if shell_name in {"bash", "zsh"}:
        return shell_name
    if os.environ.get("PSModulePath"):
        return "powershell"
    return "bash"


def _resolve_shell(value: str) -> str:
    if value in {"bash", "zsh", "powershell"}:
        return value
    return _detect_shell()


def _default_prefix(user: bool) -> Path:
    if user:
        return Path(site.getuserbase())
    return Path(sys.prefix)


def _bash_targets(prefix: Path) -> tuple[Path, ...]:
    completions_dir = prefix / "share" / "bash-completion" / "completions"
    return tuple(completions_dir / command for command in BASH_COMPLETION_COMMANDS)


def _zsh_target(prefix: Path) -> Path:
    return prefix / "share" / "zsh" / "site-functions" / "_msmodeling-python"


def _powershell_target(prefix: Path) -> Path:
    return prefix / "share" / "powershell" / "Completions" / "msmodeling.ps1"


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _completion_for_shell(shell: str) -> str:
    if shell == "zsh":
        return render_zsh_completion()
    if shell == "powershell":
        return render_powershell_completion()
    return render_bash_completion()


def _bash_rc_file() -> Path:
    return Path.home() / ".bashrc"


def _managed_block(completion_path: Path) -> str:
    quoted_path = shlex.quote(str(completion_path))
    return (
        f"{BLOCK_BEGIN}\n"
        f"if [ -r {quoted_path} ]; then\n"
        f"    . {quoted_path}\n"
        "fi\n"
        f"{BLOCK_END}\n"
    )


def _replace_managed_block(contents: str, block: str) -> str:
    lines = contents.splitlines(keepends=True)
    result: list[str] = []
    pending_block: list[str] = []
    in_block = False
    replaced = False

    for line in lines:
        stripped = line.strip()
        if stripped == BLOCK_BEGIN:
            pending_block = [line]
            in_block = True
            continue
        if in_block:
            pending_block.append(line)
        if stripped == BLOCK_END and in_block:
            if not replaced:
                if result and result[-1].strip():
                    result.append("\n")
                result.append(block)
                replaced = True
            pending_block = []
            in_block = False
            continue
        if not in_block:
            result.append(line)

    if in_block:
        result.extend(pending_block)

    if not replaced:
        if result and result[-1].strip():
            result.append("\n")
        result.append(block)

    return "".join(result)


def _remove_managed_block(contents: str) -> str:
    lines = contents.splitlines(keepends=True)
    result: list[str] = []
    pending_block: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == BLOCK_BEGIN:
            pending_block = [line]
            in_block = True
            continue
        if in_block:
            pending_block.append(line)
        if stripped == BLOCK_END and in_block:
            pending_block = []
            in_block = False
            continue
        if not in_block:
            result.append(line)
    if in_block:
        result.extend(pending_block)
    return "".join(result)


def enable_tab_completion(
    shell: str = "auto",
    rc_file: str | Path | None = None,
    completion_file: str | Path | None = None,
    reload_shell: bool = False,
) -> int:
    resolved_shell = _resolve_shell(shell)
    if resolved_shell != "bash":
        print("Only bash startup-file enablement is currently supported.", file=sys.stderr)
        print("For zsh use: . <(msmodeling-tab print --shell zsh)", file=sys.stderr)
        return 1

    completion_path = Path(completion_file) if completion_file is not None else BASH_COMPLETION_FILE
    _write(completion_path, render_bash_completion())

    target_rc = Path(rc_file) if rc_file is not None else _bash_rc_file()
    original = target_rc.read_text(encoding="utf-8") if target_rc.exists() else ""
    target_rc.write_text(_replace_managed_block(original, _managed_block(completion_path)), encoding="utf-8")

    print(f"Enabled msmodeling tab completion in {target_rc}")
    print(f"Wrote static completion script to {completion_path}")
    print("Open a new bash terminal to use it, or run: exec bash")

    if reload_shell:
        os.execvp("bash", ["bash"])
    return 0


def disable_tab_completion(
    shell: str = "auto",
    rc_file: str | Path | None = None,
    completion_file: str | Path | None = None,
    delete_completion_file: bool = False,
) -> int:
    resolved_shell = _resolve_shell(shell)
    if resolved_shell != "bash":
        print("Only bash startup-file disablement is currently supported.", file=sys.stderr)
        return 1

    target_rc = Path(rc_file) if rc_file is not None else _bash_rc_file()
    original = target_rc.read_text(encoding="utf-8") if target_rc.exists() else ""
    updated = _remove_managed_block(original)
    target_rc.write_text(updated, encoding="utf-8")

    print(f"Disabled msmodeling tab completion in {target_rc}")

    if delete_completion_file:
        completion_path = Path(completion_file) if completion_file is not None else BASH_COMPLETION_FILE
        try:
            completion_path.unlink()
            print(f"Deleted static completion script: {completion_path}")
        except FileNotFoundError:
            print(f"Static completion script already absent: {completion_path}")

    return 0


def install_completion_files(prefix: str | None = None, user: bool = False) -> int:
    target_prefix = Path(prefix) if prefix else _default_prefix(user)
    bash_paths = _bash_targets(target_prefix)
    zsh_path = _zsh_target(target_prefix)
    powershell_path = _powershell_target(target_prefix)
    bash_completion = render_bash_completion()
    for bash_path in bash_paths:
        _write(bash_path, bash_completion)
    _write(zsh_path, render_zsh_completion())
    _write(powershell_path, render_powershell_completion())
    for bash_path in bash_paths:
        print(f"Wrote bash completion: {bash_path}")
    print(f"Wrote zsh  completion: {zsh_path}")
    print(f"Wrote PowerShell completion: {powershell_path}")
    return 0


def print_completion(shell: str = "auto") -> int:
    sys.stdout.write(_completion_for_shell(_resolve_shell(shell)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit, install, or enable static shell completion for msmodeling.",
    )
    parser.add_argument(
        "--shell",
        choices=("auto", "bash", "zsh", "powershell"),
        default="auto",
        help="target shell for the print command",
    )
    subparsers = parser.add_subparsers(dest="command")

    install_parser = subparsers.add_parser("install", help="write completion files under <prefix>/share")
    install_parser.add_argument("--prefix", default=None)
    install_parser.add_argument("--user", action="store_true")

    print_parser = subparsers.add_parser("print", help="print completion script")
    print_parser.add_argument(
        "--shell",
        dest="print_shell",
        choices=("auto", "bash", "zsh", "powershell"),
        default=None,
        help="target shell; may also be passed before the subcommand",
    )
    subparsers.add_parser("enable", help="enable bash completion from ~/.bashrc")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "install":
        return install_completion_files(args.prefix, args.user)
    if args.command == "enable":
        return enable_tab_completion(args.shell)
    return print_completion(getattr(args, "print_shell", None) or args.shell)


if __name__ == "__main__":
    sys.exit(main())
