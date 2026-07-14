#!/usr/bin/env bash

linear_code_set_early_stopping_shutdown_mode() {
  local config_path="${1:?config_path is required}"
  local mode="${2:-terminate}"

  python - "$config_path" "$mode" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
mode = sys.argv[2]
if mode not in {"wait", "terminate"}:
    raise SystemExit(
        f"Invalid EARLY_STOPPING_SHUTDOWN_MODE={mode!r}; expected 'wait' or 'terminate'"
    )

text = config_path.read_text()
line = f'early_stopping_shutdown_mode: "{mode}"'
pattern = re.compile(r'(?m)^early_stopping_shutdown_mode:\s*["\']?(wait|terminate)["\']?\s*$')
if pattern.search(text):
    text = pattern.sub(line, text, count=1)
elif text.endswith("\n"):
    text = f"{text}{line}\n"
else:
    text = f"{text}\n{line}\n"
config_path.write_text(text)
PY
}

linear_code_record_sqlite() {
  local output_dir="${1:?output_dir is required}"
  local output_root="${2:-examples/linear_code_binary_search/outputs}"
  local db_path="${LINEAR_CODE_SQLITE_DB:-examples/linear_code_binary_search/code_table_records.sqlite}"
  local viewer_json="${LINEAR_CODE_VIEWER_JSON:-examples/linear_code_binary_search/code_table_viewer/code_table_data.json}"
  local record_path="${LINEAR_CODE_RECORD_PATH:-examples/linear_code_binary_search/Misc/ECCRecord.json}"
  local example_dir="examples/linear_code_binary_search"

  if [[ "${LINEAR_CODE_SQLITE_RECORD:-1}" == "0" ]]; then
    echo "Skipping SQLite record update because LINEAR_CODE_SQLITE_RECORD=0"
    return 0
  fi

  if [[ ! -d "$output_dir" ]]; then
    echo "Skipping SQLite record update: output directory does not exist: $output_dir" >&2
    return 0
  fi

  local best_program="${output_dir}/best/best_program.c"
  if [[ "${LINEAR_CODE_VERIFY_ON_RECORD:-0}" == "1" && -f "$best_program" ]]; then
    export LINEAR_CODE_N LINEAR_CODE_K LINEAR_CODE_D
    for name in \
      LINEAR_CODE_RESTARTS \
      LINEAR_CODE_RANDOM_SEED \
      LINEAR_CODE_DYNAMIC_WINDOW \
      LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE \
      LINEAR_CODE_MAX_CANDIDATES \
      LINEAR_CODE_REPAIR_EVENTS \
      LINEAR_CODE_REPAIR_DROP_COUNT \
      LINEAR_CODE_REPAIR_CANDIDATE_WINDOW \
      LINEAR_CODE_REPAIR_TABU_TENURE \
      LINEAR_CODE_REPAIR_MODE \
      LINEAR_CODE_REPAIR_MCTS_SIMULATIONS \
      LINEAR_CODE_REPAIR_MCTS_DEPTH \
      LINEAR_CODE_REPAIR_MCTS_WORKERS \
      LINEAR_CODE_REPAIR_MCTS_DROP_TOPK \
      LINEAR_CODE_C_COMPILE_TIMEOUT \
      LINEAR_CODE_C_RUN_TIMEOUT \
      LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES \
      LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP; do
      if [[ -v "$name" ]]; then
        export "$name"
      fi
    done

    python "${example_dir}/verify_distance.py" \
      --no-progress \
      "$best_program" \
      > "${output_dir}/matrix_verification.txt" \
      2> "${output_dir}/matrix_verification.stderr.log" || {
        local verify_status=$?
        echo "verify_distance.py failed with exit code ${verify_status}; importing available run metadata anyway." >&2
      }
  elif [[ "${LINEAR_CODE_VERIFY_ON_RECORD:-0}" == "1" ]]; then
    echo "No best program found at ${best_program}; importing available run metadata anyway." >&2
  fi

  python "${example_dir}/code_table_db.py" import-record \
    --db "$db_path" \
    --record "$record_path"
  python "${example_dir}/code_table_db.py" import-runs \
    --db "$db_path" \
    --search-root "$output_root"
  python "${example_dir}/code_table_db.py" export-viewer \
    --db "$db_path" \
    --output "$viewer_json"
}
