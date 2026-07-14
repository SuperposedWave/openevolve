#!/usr/bin/env bash
# Modified MCTS repair run for the [33,23,5] binary linear-code target.
#
# This script uses the updated C search skeleton version where MCTS repair:
# - compares first-drop choices with aggregated root rollout statistics;
# - uses LINEAR_CODE_REPAIR_MCTS_DROP_TOPK for stuck follow-up drops;
# - defaults to a bounded top-k follow-up drop policy instead of pure random drop.
#
# Set LINEAR_CODE_REPAIR_MCTS_DROP_TOPK=0 before running if you want to compare
# against the older random follow-up drop behavior.

set -euo pipefail

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve
cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel
. examples/linear_code_binary_search/Scripts/record_sqlite.sh

# 码长 n。
LINEAR_CODE_N=31
# 信息维度 k。
LINEAR_CODE_K=13
# 目标最小距离 d。
LINEAR_CODE_D=9
# 每次评估里 C kernel 的随机/重启搜索次数，越大越稳但越慢。
LINEAR_CODE_RESTARTS=5
# 并行执行 restart 的 worker 数。
LINEAR_CODE_RESTART_WORKERS=8
# 并行评估候选列的 worker 数。
LINEAR_CODE_CANDIDATE_WORKERS=8
# 动态候选窗口大小；对 r=10,d=5 基本覆盖全部候选，保守设为 4096。
LINEAR_CODE_DYNAMIC_WINDOW=4096
# repair 最多触发次数。
LINEAR_CODE_REPAIR_EVENTS=8
# 每次 repair 回退/删除的列数。
LINEAR_CODE_REPAIR_DROP_COUNT=1
# repair 阶段重新尝试的候选窗口大小。
LINEAR_CODE_REPAIR_CANDIDATE_WINDOW=4096
# tabu 禁忌保留轮数，避免刚删掉的列马上被选回。
LINEAR_CODE_REPAIR_TABU_TENURE=4
# repair 模式；mcts 表示使用有界局部 Monte Carlo repair。
LINEAR_CODE_REPAIR_MODE=mcts
# MCTS repair 每次触发时的 rollout/模拟次数。
LINEAR_CODE_REPAIR_MCTS_SIMULATIONS=128
# MCTS repair 每条 rollout 最多向前尝试的步数。
LINEAR_CODE_REPAIR_MCTS_DEPTH=16
# MCTS repair 并行 rollout worker 数；设为 1 保持旧串行行为，设为 0 自动使用 CPU 数，上限 64。
LINEAR_CODE_REPAIR_MCTS_WORKERS=${LINEAR_CODE_REPAIR_MCTS_WORKERS:-8}
# 修改版 MCTS：rollout 再次卡住时，按释放空间排序后从 top-k drop 中采样。
# 设为 0 可恢复旧版随机 follow-up drop。
LINEAR_CODE_REPAIR_MCTS_DROP_TOPK=${LINEAR_CODE_REPAIR_MCTS_DROP_TOPK:-2}
# native 初始化阶段最多枚举/检查的初始值数量上限。
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES=2000000000000
# 单次 C kernel 评估的超时时间，单位秒。
LINEAR_CODE_C_RUN_TIMEOUT=2400
# early stop 后的 OpenEvolve worker 关闭策略；terminate 会尽快停止评估进程，脚本仍会继续写 SQLite 记录。
EARLY_STOPPING_SHUTDOWN_MODE=${EARLY_STOPPING_SHUTDOWN_MODE:-terminate}
# OpenEvolve 总迭代次数。
ITERATIONS=120

# INITIAL_PROGRAM=examples/linear_code_binary_search/initial_program.c
INITIAL_PROGRAM=/inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel/examples/linear_code_binary_search/outputs/n31_k17_d7/mcts_topk_20260711T080043Z/best/best_program.c
CONFIG_TEMPLATE=examples/linear_code_binary_search/Configs/config_c_kernel_large.yaml
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=examples/linear_code_binary_search/outputs
TARGET_DIR=${OUTPUT_ROOT}/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}
RUN_RECORD_DIR=mcts_topk_${RUN_TAG}
OUTPUT_DIR=${TARGET_DIR}/${RUN_RECORD_DIR}
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
LINEAR_CODE_REPAIR_MODE="$LINEAR_CODE_REPAIR_MODE" \
LINEAR_CODE_REPAIR_MCTS_SIMULATIONS="$LINEAR_CODE_REPAIR_MCTS_SIMULATIONS" \
LINEAR_CODE_REPAIR_MCTS_DEPTH="$LINEAR_CODE_REPAIR_MCTS_DEPTH" \
LINEAR_CODE_REPAIR_MCTS_WORKERS="$LINEAR_CODE_REPAIR_MCTS_WORKERS" \
LINEAR_CODE_REPAIR_MCTS_DROP_TOPK="$LINEAR_CODE_REPAIR_MCTS_DROP_TOPK" \
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
