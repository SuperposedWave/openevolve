LINEAR_CODE_N=29
LINEAR_CODE_K=3
LINEAR_CODE_D=16

LINEAR_CODE_LEGALITY_ENGINE=native \
LINEAR_CODE_SEARCH_MODE=sampled_refill \
LINEAR_CODE_PROFILE=1 \
LINEAR_CODE_RESTART_WORKERS=8 \
LINEAR_CODE_RESTARTS=8 \
LINEAR_CODE_SAMPLE_POOL_SIZE=8192 \
LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL=163840 \
LINEAR_CODE_SAMPLE_MAX_REFILLS=128 \
LINEAR_CODE_SAMPLE_MAX_STALE_REFILLS=4 \
LINEAR_CODE_BACKTRACK_DEPTH=2 \
LINEAR_CODE_BACKTRACK_MAX_EVENTS=4 \
LINEAR_CODE_N=$LINEAR_CODE_N \
LINEAR_CODE_K=$LINEAR_CODE_K \
LINEAR_CODE_D=$LINEAR_CODE_D \
python openevolve-run.py \
  examples/linear_code_binary_search/initial_program.py \
  examples/linear_code_binary_search/evaluator.py \
  --config examples/linear_code_binary_search/config.yaml \
  --iterations 120 \
  --output examples/linear_code_binary_search/single_runs/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}