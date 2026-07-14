#!/usr/bin/env bash
set -euo pipefail

# 清理代理环境变量，避免本地/集群代理设置影响模型请求或依赖访问。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY ws_proxy wss_proxy WS_PROXY WSS_PROXY

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve
cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel

# 码长 n：搜索 A(n,d) 中二元码字的比特长度。
MAX_CODE_N="17"

# 最小距离 d：任意两个码字之间至少需要达到的 Hamming 距离。
MAX_CODE_D="6"

# 重启次数：同一个 priority 函数会用不同确定性 tie-break/随机种子跑几次，取最好结果。
MAX_CODE_RESTARTS="1"

# parity full-scan 每轮最多接受的中心数；只影响 d=4 parity-transform 的小规模 full-scan 路径。
MAX_CODE_PARITY_FULL_BATCH_SIZE="64"

# parity full-scan 阈值：当 d=4 且内部搜索长度 n-1 不超过该值时，使用 full-scan 动态构造。
MAX_CODE_PARITY_FULL_SCAN_N="18"

# 局部可用性采样大小：用于估计候选附近还有多少可用点，越大越准但越慢。
MAX_CODE_LOCAL_SAMPLE_SIZE="64"

# 随机种子：控制 sampled refill / MCTS rollout 的确定性随机选择，方便复现实验。
MAX_CODE_RANDOM_SEED="0"

# repair 模式：mcts 表示启用从 linear code 迁移来的 bounded rollout drop/refill 修复逻辑；本脚本默认走 C kernel。
MAX_CODE_REPAIR_MODE="mcts"

# 动态贪心窗口：C kernel 每次 refill 会在排序候选中扫描的可行候选数量；越大越准但越慢。
MAX_CODE_DYNAMIC_WINDOW="4096"

# 每个 restart 允许真实执行的 repair 事件数；C kernel 默认使用高预算，便于 A(17,6) 做更充分修复。
MAX_CODE_REPAIR_EVENTS="4"

# 每次 repair 卡住时最多连续 drop 的数量；通常设为 1，避免一次破坏太多已有结构。
MAX_CODE_REPAIR_DROP_COUNT="1"

# tabu 队列长度：最近被 drop 的码字/中心在若干 repair 步内禁止重新加入，减少来回震荡。
MAX_CODE_REPAIR_TABU_TENURE="4"

# MCTS repair 使用的候选前缀大小：rollout refill 只从排序前缀里选动态候选，越大越慢但搜索更充分。
MAX_CODE_REPAIR_CANDIDATE_WINDOW="65536"

# MCTS root-drop rollout 数：每个 root drop 的模拟次数；C kernel 默认恢复为高预算 64。
MAX_CODE_REPAIR_MCTS_SIMULATIONS="64"

# MCTS rollout 深度：一次 root drop 后最多继续 drop/refill 的层数；越深越慢但更能跳出局部卡点。
MAX_CODE_REPAIR_MCTS_DEPTH="4"

# rollout 后续 drop 的 top-k 范围：卡住后从释放空间最多的前 k 个 drop 选择；0 表示均匀随机。
MAX_CODE_REPAIR_MCTS_DROP_TOPK="2"

# C kernel 内部 MCTS worker 数：1 表示单线程确定性更稳；0 表示自动使用 CPU 数；大于 1 会并行 rollout。
MAX_CODE_REPAIR_MCTS_WORKERS="1"

# OpenEvolve 迭代次数：控制本次进化总评估轮数。
ITERATIONS="1024"

# 单次 evaluator 超时时间：C kernel 负责高预算 MCTS，仍保留 600 秒防止异常候选卡死。
EVALUATOR_TIMEOUT="600"

# OpenEvolve 并行评估数：C kernel 自己可以开 MCTS workers，这里默认 1，避免外层并行和内层线程过度竞争。
EVALUATOR_PARALLEL_EVALUATIONS="1"

# 运行时间戳：用于生成唯一输出目录，避免覆盖历史实验。
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 初始程序：只进化这个 C 文件里的 oe_max_code_priority EVOLVE-BLOCK。
INITIAL_PROGRAM="/inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel/examples/max_binary_code_search/initial_program.c"

# 配置模板：C kernel 专用 prompt；脚本会复制并替换 environment/default target 后写入输出目录。
CONFIG_TEMPLATE="examples/max_binary_code_search/Configs/config_c_kernel.yaml"

# 输出目录：保存 resolved config、日志、checkpoint 和最终程序。
OUTPUT_DIR="examples/max_binary_code_search/runs/n${MAX_CODE_N}_d${MAX_CODE_D}_mcts_${RUN_TIMESTAMP}"

# 实际运行使用的配置文件：由下面的 Python 小脚本从模板生成。
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
  "$MAX_CODE_RANDOM_SEED" \
  "$MAX_CODE_REPAIR_MODE" \
  "$MAX_CODE_DYNAMIC_WINDOW" \
  "$MAX_CODE_REPAIR_EVENTS" \
  "$MAX_CODE_REPAIR_DROP_COUNT" \
  "$MAX_CODE_REPAIR_TABU_TENURE" \
  "$MAX_CODE_REPAIR_CANDIDATE_WINDOW" \
  "$MAX_CODE_REPAIR_MCTS_SIMULATIONS" \
  "$MAX_CODE_REPAIR_MCTS_DEPTH" \
  "$MAX_CODE_REPAIR_MCTS_DROP_TOPK" \
  "$MAX_CODE_REPAIR_MCTS_WORKERS" \
  "$EVALUATOR_TIMEOUT" \
  "$EVALUATOR_PARALLEL_EVALUATIONS" <<'PY'
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
    "MAX_CODE_REPAIR_MODE": sys.argv[10],
    "MAX_CODE_DYNAMIC_WINDOW": sys.argv[11],
    "MAX_CODE_REPAIR_EVENTS": sys.argv[12],
    "MAX_CODE_REPAIR_DROP_COUNT": sys.argv[13],
    "MAX_CODE_REPAIR_TABU_TENURE": sys.argv[14],
    "MAX_CODE_REPAIR_CANDIDATE_WINDOW": sys.argv[15],
    "MAX_CODE_REPAIR_MCTS_SIMULATIONS": sys.argv[16],
    "MAX_CODE_REPAIR_MCTS_DEPTH": sys.argv[17],
    "MAX_CODE_REPAIR_MCTS_DROP_TOPK": sys.argv[18],
    "MAX_CODE_REPAIR_MCTS_WORKERS": sys.argv[19],
}
evaluator_values = {
    "timeout": sys.argv[20],
    "parallel_evaluations": sys.argv[21],
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

for key, value in evaluator_values.items():
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
MAX_CODE_REPAIR_MODE="$MAX_CODE_REPAIR_MODE" \
MAX_CODE_DYNAMIC_WINDOW="$MAX_CODE_DYNAMIC_WINDOW" \
MAX_CODE_REPAIR_EVENTS="$MAX_CODE_REPAIR_EVENTS" \
MAX_CODE_REPAIR_DROP_COUNT="$MAX_CODE_REPAIR_DROP_COUNT" \
MAX_CODE_REPAIR_TABU_TENURE="$MAX_CODE_REPAIR_TABU_TENURE" \
MAX_CODE_REPAIR_CANDIDATE_WINDOW="$MAX_CODE_REPAIR_CANDIDATE_WINDOW" \
MAX_CODE_REPAIR_MCTS_SIMULATIONS="$MAX_CODE_REPAIR_MCTS_SIMULATIONS" \
MAX_CODE_REPAIR_MCTS_DEPTH="$MAX_CODE_REPAIR_MCTS_DEPTH" \
MAX_CODE_REPAIR_MCTS_DROP_TOPK="$MAX_CODE_REPAIR_MCTS_DROP_TOPK" \
MAX_CODE_REPAIR_MCTS_WORKERS="$MAX_CODE_REPAIR_MCTS_WORKERS" \
python openevolve-run.py \
  "$INITIAL_PROGRAM" \
  examples/max_binary_code_search/evaluator.py \
  --config "$RESOLVED_CONFIG" \
  --iterations "$ITERATIONS" \
  --output "$OUTPUT_DIR"
