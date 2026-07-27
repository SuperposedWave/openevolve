#!/usr/bin/env bash
set -euo pipefail

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve
cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel
. examples/linear_code_binary_search/Scripts/record_sqlite.sh

# 码长 n。
LINEAR_CODE_N="${LINEAR_CODE_N:-50}"
# 信息维度 k；G-row 搜索会添加 k 行 P。
LINEAR_CODE_K="${LINEAR_CODE_K:-20}"
# 目标最小距离 d。
LINEAR_CODE_D="${LINEAR_CODE_D:-13}"

# 每次 evaluation 内部随机重启次数。
LINEAR_CODE_RESTARTS="${LINEAR_CODE_RESTARTS:-8}"
# 每一步最多采样多少个候选 row；越大越稳但越慢。
LINEAR_CODE_G_ROW_MAX_ATTEMPTS_PER_STEP="${LINEAR_CODE_G_ROW_MAX_ATTEMPTS_PER_STEP:-50000}"
# 每一步收集多少个已通过 hard legality 的合法 row，再交给 evolved priority 排序。
# 设为 1 是 fast randomized baseline；设为 8/16 更适合让 OpenEvolve 学排序。
LINEAR_CODE_G_ROW_LEGAL_POOL_TARGET="${LINEAR_CODE_G_ROW_LEGAL_POOL_TARGET:-8}"
# hard legality 的 near-margin 统计半径。
LINEAR_CODE_G_ROW_NEAR_MARGIN_RADIUS="${LINEAR_CODE_G_ROW_NEAR_MARGIN_RADIUS:-1}"
# repair 最多触发次数；卡住时删除若干已选 row 并重建 exact subset-xor state。
LINEAR_CODE_G_ROW_REPAIR_EVENTS="${LINEAR_CODE_G_ROW_REPAIR_EVENTS:-8}"
# 每次 repair 删除多少行。
LINEAR_CODE_G_ROW_REPAIR_DROP_COUNT="${LINEAR_CODE_G_ROW_REPAIR_DROP_COUNT:-2}"
# 删除策略：recent 删除最近行；random 随机删；tight 删除 near-margin 最紧的行。
LINEAR_CODE_G_ROW_REPAIR_STRATEGY="${LINEAR_CODE_G_ROW_REPAIR_STRATEGY:-recent}"
# repair 后被删除 row 的 tabu 保留长度，避免马上选回。
LINEAR_CODE_G_ROW_REPAIR_TABU_TENURE="${LINEAR_CODE_G_ROW_REPAIR_TABU_TENURE:-16}"
# 可选：限制随机 row 的最低 Hamming weight；默认 d-1。
LINEAR_CODE_G_ROW_MIN_ROW_WEIGHT="${LINEAR_CODE_G_ROW_MIN_ROW_WEIGHT:-}"
# 可选：priority 中偏好的 row weight；默认 r/2。
LINEAR_CODE_G_ROW_PREFER_WEIGHT="${LINEAR_CODE_G_ROW_PREFER_WEIGHT:-}"
# 随机种子。
LINEAR_CODE_RANDOM_SEED="${LINEAR_CODE_RANDOM_SEED:-1}"

# OpenEvolve evaluator 超时时间，单位秒。
EVALUATOR_TIMEOUT="${EVALUATOR_TIMEOUT:-180}"
# 外层并行 evaluation 数；G-row evaluator 单次较轻，可以适当并行。
EVALUATOR_PARALLEL_EVALUATIONS="${EVALUATOR_PARALLEL_EVALUATIONS:-4}"
# early stop 后的 OpenEvolve worker 关闭策略。
EARLY_STOPPING_SHUTDOWN_MODE="${EARLY_STOPPING_SHUTDOWN_MODE:-terminate}"
# OpenEvolve 总迭代次数。
ITERATIONS="${ITERATIONS:-80}"

INITIAL_PROGRAM="${INITIAL_PROGRAM:-examples/linear_code_binary_search/initial_program_g_row.py}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-examples/linear_code_binary_search/Configs/config_g_row.yaml}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/linear_code_binary_search/outputs}"
TARGET_DIR="${OUTPUT_ROOT}/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}"
RUN_RECORD_DIR="${RUN_RECORD_DIR:-g_row_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TARGET_DIR}/${RUN_RECORD_DIR}}"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"

mkdir -p "$OUTPUT_DIR"

python - \
  "$CONFIG_TEMPLATE" \
  "$RESOLVED_CONFIG" \
  "$LINEAR_CODE_N" \
  "$LINEAR_CODE_K" \
  "$LINEAR_CODE_D" \
  "$EVALUATOR_TIMEOUT" \
  "$EVALUATOR_PARALLEL_EVALUATIONS" <<'PY'
from pathlib import Path
import re
import sys

template_path = Path(sys.argv[1])
resolved_path = Path(sys.argv[2])
n = int(sys.argv[3])
k = int(sys.argv[4])
d = int(sys.argv[5])
evaluator_timeout = sys.argv[6]
parallel_evaluations = sys.argv[7]
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
for key, value in {
    "timeout": evaluator_timeout,
    "parallel_evaluations": parallel_evaluations,
}.items():
    key_re = re.compile(rf"(?m)^  {re.escape(key)}:\s*.*$")
    if not key_re.search(resolved_text):
        raise SystemExit(f"Failed to find evaluator key {key} in {template_path}")
    resolved_text = key_re.sub(f"  {key}: {value}", resolved_text, count=1)

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
LINEAR_CODE_N="$LINEAR_CODE_N" \
LINEAR_CODE_K="$LINEAR_CODE_K" \
LINEAR_CODE_D="$LINEAR_CODE_D" \
LINEAR_CODE_RESTARTS="$LINEAR_CODE_RESTARTS" \
LINEAR_CODE_G_ROW_MAX_ATTEMPTS_PER_STEP="$LINEAR_CODE_G_ROW_MAX_ATTEMPTS_PER_STEP" \
LINEAR_CODE_G_ROW_LEGAL_POOL_TARGET="$LINEAR_CODE_G_ROW_LEGAL_POOL_TARGET" \
LINEAR_CODE_G_ROW_NEAR_MARGIN_RADIUS="$LINEAR_CODE_G_ROW_NEAR_MARGIN_RADIUS" \
LINEAR_CODE_G_ROW_REPAIR_EVENTS="$LINEAR_CODE_G_ROW_REPAIR_EVENTS" \
LINEAR_CODE_G_ROW_REPAIR_DROP_COUNT="$LINEAR_CODE_G_ROW_REPAIR_DROP_COUNT" \
LINEAR_CODE_G_ROW_REPAIR_STRATEGY="$LINEAR_CODE_G_ROW_REPAIR_STRATEGY" \
LINEAR_CODE_G_ROW_REPAIR_TABU_TENURE="$LINEAR_CODE_G_ROW_REPAIR_TABU_TENURE" \
LINEAR_CODE_G_ROW_MIN_ROW_WEIGHT="$LINEAR_CODE_G_ROW_MIN_ROW_WEIGHT" \
LINEAR_CODE_G_ROW_PREFER_WEIGHT="$LINEAR_CODE_G_ROW_PREFER_WEIGHT" \
LINEAR_CODE_RANDOM_SEED="$LINEAR_CODE_RANDOM_SEED" \
python openevolve-run.py \
  "$INITIAL_PROGRAM" \
  examples/linear_code_binary_search/evaluator_g_row.py \
  --config "$RESOLVED_CONFIG" \
  --iterations "$ITERATIONS" \
  --output "$OUTPUT_DIR"
openevolve_status=$?
set -e

linear_code_record_sqlite "$OUTPUT_DIR" "$OUTPUT_ROOT"
exit "$openevolve_status"
