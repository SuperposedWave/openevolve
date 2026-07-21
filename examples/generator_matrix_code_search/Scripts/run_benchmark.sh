#!/usr/bin/env bash
set -euo pipefail

. /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/miniconda3/bin/activate
conda activate openevolve

cd /inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/openevolve-c-kernel

python examples/generator_matrix_code_search/search_core.py \
  --benchmark \
  --shortlist-size "${GEN_MATRIX_CODE_SHORTLIST_SIZE:-2048}"
