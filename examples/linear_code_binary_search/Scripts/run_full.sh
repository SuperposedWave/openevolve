. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate 
conda activate openevolve
export OPENAI_API_KEY=sk-c5bcdfcbc67a4652a00b0111a44ec52a

LINEAR_CODE_N=38
LINEAR_CODE_K=23
LINEAR_CODE_D=7
LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES=2000000000000


LINEAR_CODE_LEGALITY_ENGINE=native \
LINEAR_CODE_SEARCH_MODE=full \
LINEAR_CODE_PROFILE=0 \
LINEAR_CODE_CANDIDATE_EXECUTOR=process \
LINEAR_CODE_CANDIDATE_WORKERS=8 \
LINEAR_CODE_RESTART_WORKERS=8 \
LINEAR_CODE_RESTARTS=8 \
LINEAR_CODE_N=$LINEAR_CODE_N \
LINEAR_CODE_K=$LINEAR_CODE_K \
LINEAR_CODE_D=$LINEAR_CODE_D \
python openevolve-run.py \
  examples/linear_code_binary_search/initial_program.py \
  examples/linear_code_binary_search/evaluator.py \
  --config examples/linear_code_binary_search/config.yaml \
  --iterations 120 \
  --output examples/linear_code_binary_search/single_runs/n${LINEAR_CODE_N}_k${LINEAR_CODE_K}_d${LINEAR_CODE_D}