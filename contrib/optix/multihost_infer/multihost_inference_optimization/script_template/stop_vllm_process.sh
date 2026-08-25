#!/bin/bash
# -------------------------------------------------------------------------
# Leftover process cleanup script
#
# Input convention (both channels carry the same value; use whichever you prefer):
#   - the first argument $1
#   - the environment variable MODEL_EVAL_STATE_KILL_PROCESS_NAME
#   Both are comma-separated process name strings, for example:
#       "mindie, mindie-llm, mindieservice_daemon, mindie_llm"
#   A comma may be followed by spaces, so the script trims surrounding whitespace itself.
#
# Note: this script is run by bash (no executable bit or shebang required), and any
#       non-zero exit code is only recorded in the log; it never interrupts the main flow.
# -------------------------------------------------------------------------
set -u

names="${1:-${MODEL_EVAL_STATE_KILL_PROCESS_NAME:-}}"
if [ -z "${names}" ]; then
    echo "no process name provided, skip"
    exit 0
fi

# Maximum number of retries, to prevent an infinite loop
MAX_RETRY=5
# Wait time (seconds) after each kill, giving the driver time to reclaim NPU memory
WAIT_SECONDS=2

# Split on commas and kill each one in turn
old_ifs="${IFS}"
IFS=','
for raw in ${names}; do
    IFS="${old_ifs}"
    # Trim surrounding whitespace
    name="$(echo "${raw}" | xargs)"
    [ -z "${name}" ] && continue

    retry=0
    while [ ${retry} -lt ${MAX_RETRY} ]; do
        # -i ignores case (so vllm also matches VLLM::EngineCore); -f matches the full command line
        if ! pgrep -i -f "${name}" > /dev/null 2>&1; then
            echo "no residual process matching: ${name}"
            break
        fi
        echo "killing residual process matching: ${name} (attempt $((retry + 1))/${MAX_RETRY})"
        pkill -9 -i -f "${name}" || true
        sleep ${WAIT_SECONDS}
        retry=$((retry + 1))
    done

    if [ ${retry} -ge ${MAX_RETRY} ]; then
        echo "WARNING: process matching '${name}' still exists after ${MAX_RETRY} attempts"
    fi
done
IFS="${old_ifs}"

exit 0
