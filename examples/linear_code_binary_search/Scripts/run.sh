#!/usr/bin/env bash
set -euo pipefail

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve
cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel
. examples/linear_code_binary_search/Scripts/record_sqlite.sh

LINEAR_CODE_N="${LINEAR_CODE_N:-27}"
LINEAR_CODE_K="${LINEAR_CODE_K:-13}"
LINEAR_CODE_D="${LINEAR_CODE_D:-8}"
LINEAR_CODE_RESTARTS="${LINEAR_CODE_RESTARTS:-8}"
LINEAR_CODE_RESTART_WORKERS="${LINEAR_CODE_RESTART_WORKERS:-8}"
LINEAR_CODE_CANDIDATE_WORKERS="${LINEAR_CODE_CANDIDATE_WORKERS:-8}"
LINEAR_CODE_DYNAMIC_WINDOW="${LINEAR_CODE_DYNAMIC_WINDOW:-4096}"
LINEAR_CODE_REPAIR_EVENTS="${LINEAR_CODE_REPAIR_EVENTS:-4}"
LINEAR_CODE_REPAIR_DROP_COUNT="${LINEAR_CODE_REPAIR_DROP_COUNT:-1}"
LINEAR_CODE_REPAIR_CANDIDATE_WINDOW="${LINEAR_CODE_REPAIR_CANDIDATE_WINDOW:-65536}"
LINEAR_CODE_REPAIR_TABU_TENURE="${LINEAR_CODE_REPAIR_TABU_TENURE:-4}"
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES="${LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES:-2000000000000}"
LINEAR_CODE_C_RUN_TIMEOUT="${LINEAR_CODE_C_RUN_TIMEOUT:-2400}"
LINEAR_CODE_LEGALITY_ENGINE="${LINEAR_CODE_LEGALITY_ENGINE:-h_forbidden}"
LINEAR_CODE_G_VERIFY_MAX_K="${LINEAR_CODE_G_VERIFY_MAX_K:-24}"
EARLY_STOPPING_SHUTDOWN_MODE="${EARLY_STOPPING_SHUTDOWN_MODE:-terminate}"
ITERATIONS="${ITERATIONS:-120}"

INITIAL_PROGRAM="${INITIAL_PROGRAM:-examples/linear_code_binary_search/initial_program.c}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-examples/linear_code_binary_search/Configs/config_c_kernel.yaml}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/linear_code_binary_search/outputs}"
TARGET_DIR="${OUTPUT_ROOT}/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}"
RUN_RECORD_DIR="${RUN_RECORD_DIR:-run_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TARGET_DIR}/${RUN_RECORD_DIR}}"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"

mkdir -p "$OUTPUT_DIR"

python - "$CONFIG_TEMPLATE" "$RESOLVED_CONFIG" "$LINEAR_CODE_N" "$LINEAR_CODE_K" "$LINEAR_CODE_D" <<'PY'
from pathlib import Path
import re
import sys

template_path = Path(sys.argv[1])
resolved_path = Path(sys.argv[2])
n = int(sys.argv[3])
k = int(sys.argv[4])
d = int(sys.argv[5])
r = n - k

if r <= 0:
    raise SystemExit(f"Invalid linear-code target: requires n > k, got n={n}, k={k}")

target_block_re = re.compile(
    r"(?m)^    Current target:\n(?:^    - .*\n){4}"
)
replacement = (
    "    Current target:\n"
    f"    - n = {n}\n"
    f"    - k = {k}\n"
    f"    - d = {d}\n"
    f"    - r = n - k = {r}\n"
)

template_text = template_path.read_text()
if not target_block_re.search(template_text):
    raise SystemExit(f"Failed to find the 'Current target' block in {template_path}")

resolved_text = target_block_re.sub(replacement, template_text, count=1)
llm_path_re = re.compile(r"(?m)^(llm_config_path:\s*)[\"']?([^\"'\n]+)[\"']?\s*$")

def resolve_llm_path(match):
    llm_path = Path(match.group(2)).expanduser()
    if not llm_path.is_absolute():
        llm_path = (template_path.parent / llm_path).resolve()
    return f'{match.group(1)}"{llm_path}"'

resolved_path.write_text(llm_path_re.sub(resolve_llm_path, resolved_text, count=1))
PY

linear_code_set_early_stopping_shutdown_mode "$RESOLVED_CONFIG" "$EARLY_STOPPING_SHUTDOWN_MODE"

set +e
LINEAR_CODE_PROFILE=1 \
LINEAR_CODE_LEGALITY_ENGINE="$LINEAR_CODE_LEGALITY_ENGINE" \
LINEAR_CODE_G_VERIFY_MAX_K="$LINEAR_CODE_G_VERIFY_MAX_K" \
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES="$LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES" \
LINEAR_CODE_CANDIDATE_EXECUTOR=process \
LINEAR_CODE_C_RUN_TIMEOUT="$LINEAR_CODE_C_RUN_TIMEOUT" \
LINEAR_CODE_CANDIDATE_WORKERS="$LINEAR_CODE_CANDIDATE_WORKERS" \
LINEAR_CODE_RESTART_WORKERS="$LINEAR_CODE_RESTART_WORKERS" \
LINEAR_CODE_RESTARTS="$LINEAR_CODE_RESTARTS" \
LINEAR_CODE_DYNAMIC_WINDOW="$LINEAR_CODE_DYNAMIC_WINDOW" \
LINEAR_CODE_REPAIR_EVENTS="$LINEAR_CODE_REPAIR_EVENTS" \
LINEAR_CODE_REPAIR_DROP_COUNT="$LINEAR_CODE_REPAIR_DROP_COUNT" \
LINEAR_CODE_REPAIR_CANDIDATE_WINDOW="$LINEAR_CODE_REPAIR_CANDIDATE_WINDOW" \
LINEAR_CODE_REPAIR_TABU_TENURE="$LINEAR_CODE_REPAIR_TABU_TENURE" \
LINEAR_CODE_N="$LINEAR_CODE_N" \
LINEAR_CODE_K="$LINEAR_CODE_K" \
LINEAR_CODE_D="$LINEAR_CODE_D" \
python openevolve-run.py \
  "$INITIAL_PROGRAM" \
  examples/linear_code_binary_search/evaluator.py \
  --config "$RESOLVED_CONFIG" \
  --iterations "$ITERATIONS" \
  --output "$OUTPUT_DIR"
openevolve_status=$?
set -e

linear_code_record_sqlite "$OUTPUT_DIR" "$OUTPUT_ROOT"
exit "$openevolve_status"
