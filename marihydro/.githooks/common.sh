#!/bin/sh
set -eu

hook_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$hook_dir/.." && pwd)

run_powershell_script() {
    script_path=$1
    shift

    if command -v pwsh >/dev/null 2>&1; then
        exec pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$script_path" "$@"
    fi

    exec powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$script_path" "$@"
}
