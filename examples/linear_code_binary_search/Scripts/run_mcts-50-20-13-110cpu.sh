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

# 单次 evaluation 内部重启次数；进化阶段先用 8，优先提高可评估 priority 的数量。
LINEAR_CODE_RESTARTS=1
# 并行执行 restart 的 worker 数；该值超过 RESTARTS 会被 C kernel 自动 cap，所以这里设为 8。
LINEAR_CODE_RESTART_WORKERS=1
# 允许大候选集也启用 restart 并行；默认保护阈值是 2000000，这里显式放开到 1e9。
LINEAR_CODE_RESTART_PARALLEL_MAX_CANDIDATES=1000000000

# 候选上限；0 表示不采样，使用完整候选集合。n=50,k=20,d=13 的候选约 9.66e8。
LINEAR_CODE_MAX_CANDIDATES=0
# 并行评估初始候选 priority 的线程数；候选评分只发生一次，32 通常比 110 更稳。
LINEAR_CODE_CANDIDATE_WORKERS=32

# 动态候选窗口大小；完整候选+d=13 很重，先用 64 保持吞吐。
LINEAR_CODE_DYNAMIC_WINDOW=64
# 动态窗口内部并行 worker 数；每一步会并行评分窗口里的合法候选。
LINEAR_CODE_DYNAMIC_WORKERS=16
# 是否计算 exact forbidden growth；0 表示关闭。d=13 下开启会极慢，建议先关闭做高吞吐搜索。
LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE=0
# growth scratch touched-list 上限；growth 关闭时基本不生效，保留用于后续开关实验。
LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP=65536

# repair 模式；mcts 表示使用有界局部 Monte Carlo repair。
LINEAR_CODE_REPAIR_MODE=mcts
# repair 最多触发次数；高并行完整候选先控制在 2，避免每个 restart 内部过重。
LINEAR_CODE_REPAIR_EVENTS=2
# 每次 repair 删除的列数；建议先 1，减少破坏性并降低 rebuild 成本。
LINEAR_CODE_REPAIR_DROP_COUNT=1
# repair 阶段重新尝试的候选窗口大小；越大越充分但越慢。
LINEAR_CODE_REPAIR_CANDIDATE_WINDOW=4096
# tabu 禁忌保留轮数，避免刚删掉的列马上被选回。
LINEAR_CODE_REPAIR_TABU_TENURE=4
# MCTS repair 每次触发时的 rollout/模拟次数。
LINEAR_CODE_REPAIR_MCTS_SIMULATIONS=16
# MCTS repair 每条 rollout 最多向前尝试的步数。
LINEAR_CODE_REPAIR_MCTS_DEPTH=2
# MCTS 内部 worker 数；restart 已经并行，这里保持 1，避免线程数爆炸。
LINEAR_CODE_REPAIR_MCTS_WORKERS=8
# 后续 drop 从释放空间最多的前 k 个里采样；2 是较稳的默认值。
LINEAR_CODE_REPAIR_MCTS_DROP_TOPK=2

# native 初始化阶段最多枚举/检查的初始值数量上限。
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES=2000000000000
# 单次 C kernel evaluation 超时时间，单位秒。
LINEAR_CODE_C_RUN_TIMEOUT=24000
# OpenEvolve evaluator 超时时间，单位秒；和 C run timeout 对齐。
EVALUATOR_TIMEOUT=24000
# OpenEvolve 外层并行 evaluation 数；C kernel 内部已经高度并行，外层保持 1。
EVALUATOR_PARALLEL_EVALUATIONS=1
# 是否打印每次 evaluation 的详细进度和阶段耗时；1 开启，0 关闭，可在运行脚本前用环境变量覆盖。
LINEAR_CODE_VERBOSE_PROGRESS="${LINEAR_CODE_VERBOSE_PROGRESS:-1}"
# early stop 后的 OpenEvolve worker 关闭策略；terminate 会尽快停止评估进程，脚本仍会继续写 SQLite 记录。
EARLY_STOPPING_SHUTDOWN_MODE=${EARLY_STOPPING_SHUTDOWN_MODE:-terminate}
# OpenEvolve 总迭代次数。
ITERATIONS=120

INITIAL_PROGRAM=examples/linear_code_binary_search/outputs/n37_k22_d7/mcts_20260629T142425Z/best/best_program.c
CONFIG_TEMPLATE=examples/linear_code_binary_search/Configs/config_c_kernel_large.yaml
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=examples/linear_code_binary_search/outputs
TARGET_DIR=${OUTPUT_ROOT}/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}
RUN_RECORD_DIR=mcts_110cpu_${RUN_TAG}
OUTPUT_DIR=${TARGET_DIR}/${RUN_RECORD_DIR}
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

resolved_path.write_text(llm_path_re.sub(resolve_llm_path, resolved_text))
PY

linear_code_set_early_stopping_shutdown_mode "$RESOLVED_CONFIG" "$EARLY_STOPPING_SHUTDOWN_MODE"

set +e
LINEAR_CODE_PROFILE=1 \
LINEAR_CODE_VERBOSE_PROGRESS="$LINEAR_CODE_VERBOSE_PROGRESS" \
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES="$LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES" \
LINEAR_CODE_CANDIDATE_EXECUTOR=process \
LINEAR_CODE_MAX_CANDIDATES="$LINEAR_CODE_MAX_CANDIDATES" \
LINEAR_CODE_C_RUN_TIMEOUT="$LINEAR_CODE_C_RUN_TIMEOUT" \
LINEAR_CODE_CANDIDATE_WORKERS="$LINEAR_CODE_CANDIDATE_WORKERS" \
LINEAR_CODE_RESTART_WORKERS="$LINEAR_CODE_RESTART_WORKERS" \
LINEAR_CODE_RESTART_PARALLEL_MAX_CANDIDATES="$LINEAR_CODE_RESTART_PARALLEL_MAX_CANDIDATES" \
LINEAR_CODE_RESTARTS="$LINEAR_CODE_RESTARTS" \
LINEAR_CODE_DYNAMIC_WINDOW="$LINEAR_CODE_DYNAMIC_WINDOW" \
LINEAR_CODE_DYNAMIC_WORKERS="$LINEAR_CODE_DYNAMIC_WORKERS" \
LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE="$LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE" \
LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP="$LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP" \
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
