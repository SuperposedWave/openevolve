#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY ws_proxy wss_proxy WS_PROXY WSS_PROXY

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve
cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel

MAX_CODE_N="17"
MAX_CODE_D="6"
MAX_CODE_RESTARTS="1"
MAX_CODE_PARITY_FULL_BATCH_SIZE="64"
MAX_CODE_PARITY_FULL_SCAN_N="18"
MAX_CODE_LOCAL_SAMPLE_SIZE="64"
MAX_CODE_RANDOM_SEED="0"
ITERATIONS="1024"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

INITIAL_PROGRAM="/inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel/examples/max_binary_code_search/initial_program.py"
# INITIAL_PROGRAM="examples/max_binary_code_search/initial_program.py"
CONFIG_TEMPLATE="examples/max_binary_code_search/Configs/config.yaml"
OUTPUT_DIR="examples/max_binary_code_search/runs/n${MAX_CODE_N}_d${MAX_CODE_D}_parity_dynamic_${RUN_TIMESTAMP}"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty output directory: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python - \
  "$CONFIG_TEMPLATE" \
  "$RESOLVED_CONFIG" \
  "$MAX_CODE_N" \
  "$MAX_CODE_D" \
  "$MAX_CODE_RESTARTS" \
  "$MAX_CODE_PARITY_FULL_BATCH_SIZE" \
  "$MAX_CODE_PARITY_FULL_SCAN_N" \
  "$MAX_CODE_LOCAL_SAMPLE_SIZE" \
  "$MAX_CODE_RANDOM_SEED" <<'PY'
from pathlib import Path
import re
import sys

template_path = Path(sys.argv[1])
resolved_path = Path(sys.argv[2])
n = int(sys.argv[3])
d = int(sys.argv[4])
environment_values = {
    "MAX_CODE_N": sys.argv[3],
    "MAX_CODE_D": sys.argv[4],
    "MAX_CODE_RESTARTS": sys.argv[5],
    "MAX_CODE_PARITY_FULL_BATCH_SIZE": sys.argv[6],
    "MAX_CODE_PARITY_FULL_SCAN_N": sys.argv[7],
    "MAX_CODE_LOCAL_SAMPLE_SIZE": sys.argv[8],
    "MAX_CODE_RANDOM_SEED": sys.argv[9],
}

target_block_re = re.compile(
    r"(?m)^    Current default target:\n(?:^    - .*\n){2}"
)
replacement = (
    "    Current default target:\n"
    f"    - n = {n}\n"
    f"    - d = {d}\n"
)

template_text = template_path.read_text()
if not target_block_re.search(template_text):
    raise SystemExit(f"Failed to find the 'Current default target' block in {template_path}")

resolved_text = target_block_re.sub(replacement, template_text, count=1)

for key, value in environment_values.items():
    key_re = re.compile(rf"(?m)^  {re.escape(key)}:\s*.*$")
    if not key_re.search(resolved_text):
        raise SystemExit(f"Failed to find environment key {key} in {template_path}")
    resolved_text = key_re.sub(f'  {key}: "{value}"', resolved_text, count=1)

llm_path_re = re.compile(r"(?m)^(llm_config_path:\s*)[\"']?([^\"'\n]+)[\"']?\s*$")

def resolve_llm_path(match):
    llm_path = Path(match.group(2)).expanduser()
    if not llm_path.is_absolute():
        llm_path = (template_path.parent / llm_path).resolve()
    return f'{match.group(1)}"{llm_path}"'

resolved_text = llm_path_re.sub(resolve_llm_path, resolved_text)
resolved_path.write_text(resolved_text)
PY

MAX_CODE_N="$MAX_CODE_N" \
MAX_CODE_D="$MAX_CODE_D" \
MAX_CODE_RESTARTS="$MAX_CODE_RESTARTS" \
MAX_CODE_PARITY_FULL_BATCH_SIZE="$MAX_CODE_PARITY_FULL_BATCH_SIZE" \
MAX_CODE_PARITY_FULL_SCAN_N="$MAX_CODE_PARITY_FULL_SCAN_N" \
MAX_CODE_LOCAL_SAMPLE_SIZE="$MAX_CODE_LOCAL_SAMPLE_SIZE" \
MAX_CODE_RANDOM_SEED="$MAX_CODE_RANDOM_SEED" \
python openevolve-run.py \
  "$INITIAL_PROGRAM" \
  examples/max_binary_code_search/evaluator.py \
  --config "$RESOLVED_CONFIG" \
  --iterations "$ITERATIONS" \
  --output "$OUTPUT_DIR" 
#   \
#   --checkpoint examples/max_binary_code_search/runs/n17_d6_parity_dynamic_20260621_063321/checkpoints/checkpoint_1020
