#!/usr/bin/env bash
set -euo pipefail

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve
cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel
. examples/linear_code_binary_search/Scripts/record_sqlite.sh

# 码长 n。
LINEAR_CODE_N=50
# 信息维度 k。
LINEAR_CODE_K=20
# 目标最小距离 d。
LINEAR_CODE_D=13
# 每次评估里 C kernel 的随机/重启搜索次数；该目标单次 restart 已经较重，先用 1。
LINEAR_CODE_RESTARTS=1
# 并行执行 restart 的 worker 数。
LINEAR_CODE_RESTART_WORKERS=8
# 并行评估候选列的 worker 数。
LINEAR_CODE_CANDIDATE_WORKERS=8
# 单次 evaluate 实际打分/排序的候选列上限；n50/k20/d13 理论候选约 9.66 亿，必须采样。
LINEAR_CODE_MAX_CANDIDATES=1000
# 动态候选窗口大小；此脚本保留 dynamic 选择，但关闭 exact growth 估计。
LINEAR_CODE_DYNAMIC_WINDOW=64
# 是否计算动态候选的 exact forbidden growth；0 表示不计算，避免 d=13 下的高成本。
LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE=3
# repair 最多触发次数；当前 d=13 的 MCTS rollout/rebuild 太重，先关闭 repair。
LINEAR_CODE_REPAIR_EVENTS=0
# 每次 repair 回退/删除的最近列数；repair 关闭时该参数不会生效。
LINEAR_CODE_REPAIR_DROP_COUNT=3
# repair 阶段重新尝试的候选窗口大小；repair 关闭时该参数不会生效。
LINEAR_CODE_REPAIR_CANDIDATE_WINDOW=65536
# tabu 禁忌保留轮数；repair 关闭时该参数不会生效。
LINEAR_CODE_REPAIR_TABU_TENURE=4
# repair 模式；repair 关闭时该参数不会生效。
LINEAR_CODE_REPAIR_MODE=mcts
# MCTS repair 每次触发时的 rollout/模拟次数；repair 关闭时该参数不会生效。
LINEAR_CODE_REPAIR_MCTS_SIMULATIONS=1
# MCTS repair 每条 rollout 最多向前尝试的步数；repair 关闭时该参数不会生效。
LINEAR_CODE_REPAIR_MCTS_DEPTH=1
# 动态 growth 估计 scratch touched-list 上限；growth estimate 关闭时该参数不会生效。
LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP=65536
# legality engine: h_forbidden 保持旧 H 禁区；g_layers 使用 G=[I_k|P] 的 subset-xor 层判定。
LINEAR_CODE_LEGALITY_ENGINE="${LINEAR_CODE_LEGALITY_ENGINE:-h_forbidden}"
# G 侧 Gray-code exact verification 的最大 filled k；n50/k20 会完整验证。
LINEAR_CODE_G_VERIFY_MAX_K="${LINEAR_CODE_G_VERIFY_MAX_K:-24}"
# native 初始化阶段最多枚举/检查的初始值数量上限。
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES=2000000000000
# 单次 C kernel 评估的超时时间，单位秒。
LINEAR_CODE_C_RUN_TIMEOUT=180
# early stop 后的 OpenEvolve worker 关闭策略；terminate 会尽快停止评估进程，脚本仍会继续写 SQLite 记录。
EARLY_STOPPING_SHUTDOWN_MODE=${EARLY_STOPPING_SHUTDOWN_MODE:-terminate}
# OpenEvolve 总迭代次数。
ITERATIONS=120

INITIAL_PROGRAM=examples/linear_code_binary_search/outputs/n37_k22_d7/mcts_20260629T142425Z/best/best_program.c
CONFIG_TEMPLATE=examples/linear_code_binary_search/Configs/config_c_kernel_large.yaml
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=examples/linear_code_binary_search/outputs
TARGET_DIR=${OUTPUT_ROOT}/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}
RUN_RECORD_DIR=dynamic_nogrowth_${RUN_TAG}
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
LINEAR_CODE_LEGALITY_ENGINE="$LINEAR_CODE_LEGALITY_ENGINE" \
LINEAR_CODE_G_VERIFY_MAX_K="$LINEAR_CODE_G_VERIFY_MAX_K" \
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES="$LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES" \
LINEAR_CODE_CANDIDATE_EXECUTOR=process \
LINEAR_CODE_MAX_CANDIDATES="$LINEAR_CODE_MAX_CANDIDATES" \
LINEAR_CODE_C_RUN_TIMEOUT="$LINEAR_CODE_C_RUN_TIMEOUT" \
LINEAR_CODE_CANDIDATE_WORKERS="$LINEAR_CODE_CANDIDATE_WORKERS" \
LINEAR_CODE_RESTART_WORKERS="$LINEAR_CODE_RESTART_WORKERS" \
LINEAR_CODE_RESTARTS="$LINEAR_CODE_RESTARTS" \
LINEAR_CODE_DYNAMIC_WINDOW="$LINEAR_CODE_DYNAMIC_WINDOW" \
LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE="$LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE" \
LINEAR_CODE_REPAIR_EVENTS="$LINEAR_CODE_REPAIR_EVENTS" \
LINEAR_CODE_REPAIR_DROP_COUNT="$LINEAR_CODE_REPAIR_DROP_COUNT" \
LINEAR_CODE_REPAIR_CANDIDATE_WINDOW="$LINEAR_CODE_REPAIR_CANDIDATE_WINDOW" \
LINEAR_CODE_REPAIR_TABU_TENURE="$LINEAR_CODE_REPAIR_TABU_TENURE" \
LINEAR_CODE_REPAIR_MODE="$LINEAR_CODE_REPAIR_MODE" \
LINEAR_CODE_REPAIR_MCTS_SIMULATIONS="$LINEAR_CODE_REPAIR_MCTS_SIMULATIONS" \
LINEAR_CODE_REPAIR_MCTS_DEPTH="$LINEAR_CODE_REPAIR_MCTS_DEPTH" \
LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP="$LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP" \
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
