#!/usr/bin/env bash
set -euo pipefail

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve

cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel

: "${GEN_MATRIX_CODE_N:=50}"
: "${GEN_MATRIX_CODE_K:=20}"
: "${GEN_MATRIX_CODE_D:=13}"
: "${GEN_MATRIX_CODE_RESTARTS:=1}"
: "${GEN_MATRIX_CODE_SHORTLIST_SIZE:=1024}"
: "${GEN_MATRIX_CODE_RANDOM_SEED:=1}"
: "${GEN_MATRIX_CODE_CONFIG:=examples/generator_matrix_code_search/Configs/config.yaml}"
: "${GEN_MATRIX_CODE_ITERATIONS:=40}"
: "${GEN_MATRIX_CODE_RESOLVED_CONFIG:=examples/generator_matrix_code_search/Configs/.resolved_config.yaml}"

export GEN_MATRIX_CODE_N
export GEN_MATRIX_CODE_K
export GEN_MATRIX_CODE_D
export GEN_MATRIX_CODE_RESTARTS
export GEN_MATRIX_CODE_SHORTLIST_SIZE
export GEN_MATRIX_CODE_RANDOM_SEED

python examples/generator_matrix_code_search/render_config.py \
  "$GEN_MATRIX_CODE_CONFIG" \
  "$GEN_MATRIX_CODE_RESOLVED_CONFIG" \
  --N "$GEN_MATRIX_CODE_N" \
  --K "$GEN_MATRIX_CODE_K" \
  --D "$GEN_MATRIX_CODE_D"

python openevolve-run.py \
  examples/generator_matrix_code_search/initial_program.py \
  examples/generator_matrix_code_search/evaluator.py \
  --config "$GEN_MATRIX_CODE_RESOLVED_CONFIG" \
  --iterations "$GEN_MATRIX_CODE_ITERATIONS"
