#!/usr/bin/env bash
set -euo pipefail

# JIT Remote Cache test harness for Qwen3.5-35B-A3B-FP8, TP4 (4x H20).
# Covers prefill-only and decode-heavy prompt lengths with repeat=5 per length
# so p50/p99 can be computed. Designed to be called once per scenario
# (snapshot_warmup / direct_warmup / snapshot_warm / direct_warm / local_warm).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" && -x "/usr/local/cuda-12.9/bin/nvcc" ]]; then
  CUDA_HOME="/usr/local/cuda-12.9"
fi
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"

MODEL_TYPE="${MODEL_TYPE:-qwen35_moe}"
MODEL_PATH="${MODEL_PATH:-/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8}"
# Comma-separated GPU indices for TP4 (4 GPUs needed)
GPUS="${GPUS:-2,3,4,5}"
TP_SIZE="${TP_SIZE:-4}"

REMOTE_JIT_DIR="${REMOTE_JIT_DIR:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/data1/ganyiwei.gyw/RTP-LLM/jit_cache_results/${RUN_ID}}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/qwen35_moe_jit_workloads}"
LOCAL_JIT_DIR="${LOCAL_JIT_DIR:-${RESULT_DIR}/local_jit}"

# Prompt lengths list (prefill group, max_new_tokens=1, repeat=5)
PROMPT_WORDS_LIST="${PROMPT_WORDS_LIST:-64,128,192,256,384,512,768,1024,1536,2048}"
REPEAT_PER_LENGTH="${REPEAT_PER_LENGTH:-5}"
# Short decode group (max_new_tokens=128, repeat=3), same lengths
DECODE_PROMPT_WORDS_LIST="${DECODE_PROMPT_WORDS_LIST:-64,128,256,512,1024}"
DECODE_REPEAT="${DECODE_REPEAT:-3}"
DECODE_MAX_NEW_TOKENS="${DECODE_MAX_NEW_TOKENS:-128}"

SMOKE_ARGS="${SMOKE_ARGS:---warm_up 0 \
  --seq_size_per_block 64 \
  --act_type BF16 \
  --reserver_runtime_mem_mb 8192 \
  --enable_cuda_graph 0 \
  --reuse_cache 1 \
  --tp_size ${TP_SIZE} \
  --world_size ${TP_SIZE} \
  --dp_size 1 \
  --fp8_kv_cache 1 \
  --load_method scratch}"

if [[ -z "${REMOTE_JIT_DIR}" ]]; then
  echo "REMOTE_JIT_DIR is required, for example:" >&2
  echo "  REMOTE_JIT_DIR=/mnt/nas1/rtp_llm_jit_cache/snapshot/\${RUN_ID} $0" >&2
  exit 2
fi
if [[ ! -x "${NVCC_BIN}" ]]; then
  echo "nvcc not found: ${NVCC_BIN}" >&2
  exit 2
fi

mkdir -p "${REMOTE_JIT_DIR}" "${LOCAL_JIT_DIR}" "${RESULT_DIR}"

PROTO_DIR="${REPO_ROOT}/rtp_llm/cpp/model_rpc/proto"
PROTO_GEN_DIR="${PROTO_GEN_DIR:-/tmp/rtp_proto_gen_check/rtp_llm/cpp/model_rpc/proto}"
for f in model_rpc_service_pb2.py model_rpc_service_pb2_grpc.py; do
  if [[ ! -e "${PROTO_DIR}/${f}" && -e "${PROTO_GEN_DIR}/${f}" ]]; then
    ln -s "${PROTO_GEN_DIR}/${f}" "${PROTO_DIR}/${f}"
  fi
done

# Build combined task JSON: prefill group (max_new_tokens=1) + decode group (max_new_tokens=128)
TASK_INFO="${RESULT_DIR}/qwen35_moe_jit_workloads.json"
export TASK_INFO MODEL_TYPE MODEL_PATH
export PROMPT_WORDS_LIST REPEAT_PER_LENGTH
export DECODE_PROMPT_WORDS_LIST DECODE_REPEAT DECODE_MAX_NEW_TOKENS
"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path

def words_list(env_key):
    return [int(x.strip()) for x in os.environ[env_key].split(",") if x.strip()]

def make_prompt(words):
    return " ".join(["hello"] * words)

prefill_lens = words_list("PROMPT_WORDS_LIST")
prefill_repeat = int(os.environ["REPEAT_PER_LENGTH"])
decode_lens = words_list("DECODE_PROMPT_WORDS_LIST")
decode_repeat = int(os.environ["DECODE_REPEAT"])
decode_max_new = int(os.environ["DECODE_MAX_NEW_TOKENS"])

query_result = []

for words in prefill_lens:
    for i in range(prefill_repeat):
        query_result.append({
            "query": {
                "prompt": [make_prompt(words)],
                "generate_config": {"max_new_tokens": 1, "top_k": 1, "top_p": 0},
            },
            "result": {"response": "", "finished": True},
            "_jit_workload": {
                "group": "prefill",
                "name": f"prefill_len_{words}",
                "repeat_index": i + 1,
                "prompt_words": words,
                "max_new_tokens": 1,
            },
        })

for words in decode_lens:
    for i in range(decode_repeat):
        query_result.append({
            "query": {
                "prompt": [make_prompt(words)],
                "generate_config": {"max_new_tokens": decode_max_new, "top_k": 1, "top_p": 0},
            },
            "result": {"response": "", "finished": True},
            "_jit_workload": {
                "group": "decode",
                "name": f"decode_len_{words}",
                "repeat_index": i + 1,
                "prompt_words": words,
                "max_new_tokens": decode_max_new,
            },
        })

task = {
    "model_type": os.environ["MODEL_TYPE"],
    "model_path": os.environ["MODEL_PATH"],
    "query_result": query_result,
}
Path(os.environ["TASK_INFO"]).write_text(
    json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"task_info written: {os.environ['TASK_INFO']} ({len(query_result)} queries)")
PY

cat > "${RESULT_DIR}/manifest.txt" <<EOF
run_id=${RUN_ID}
repo_root=${REPO_ROOT}
model_type=${MODEL_TYPE}
model_path=${MODEL_PATH}
gpus=${GPUS}
tp_size=${TP_SIZE}
remote_jit_dir=${REMOTE_JIT_DIR}
local_jit_dir=${LOCAL_JIT_DIR}
cuda_home=${CUDA_HOME}
nvcc_bin=${NVCC_BIN}
result_dir=${RESULT_DIR}
task_info=${TASK_INFO}
prompt_words_list=${PROMPT_WORDS_LIST}
repeat_per_length=${REPEAT_PER_LENGTH}
decode_prompt_words_list=${DECODE_PROMPT_WORDS_LIST}
decode_repeat=${DECODE_REPEAT}
decode_max_new_tokens=${DECODE_MAX_NEW_TOKENS}
smoke_args=${SMOKE_ARGS}
EOF

export TEST_UNDECLARED_OUTPUTS_DIR="${RESULT_DIR}"
export REMOTE_JIT_DIR LOCAL_JIT_DIR RESULT_DIR
export JIT_CACHE_DETAILED_STATS=1
export SAVE_RESPONSE=True
export ACCL_LOW_LATENCY_OPTIMIZE=1
export CUDA_VISIBLE_DEVICES="${GPUS}"
export LOCAL_WORLD_SIZE="${TP_SIZE}"
export CUDA_HOME
export TRTLLM_DG_NVCC_COMPILER="${TRTLLM_DG_NVCC_COMPILER:-${NVCC_BIN}}"
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/rtp_llm/test:${PYTHONPATH:-}"

ENV_JSON="$(TASK_ENV_REMOTE="${REMOTE_JIT_DIR}" TASK_ENV_LOCAL="${LOCAL_JIT_DIR}" TASK_ENV_GPUS="${GPUS}" TASK_ENV_TP="${TP_SIZE}" "${PYTHON_BIN}" - <<'PY'
import json, os
print(json.dumps([
    f"REMOTE_JIT_DIR={os.environ['TASK_ENV_REMOTE']}",
    f"LOCAL_JIT_DIR={os.environ['TASK_ENV_LOCAL']}",
    "JIT_CACHE_DETAILED_STATS=1",
    "SAVE_RESPONSE=True",
    "ACCL_LOW_LATENCY_OPTIMIZE=1",
    f"CUDA_VISIBLE_DEVICES={os.environ['TASK_ENV_GPUS']}",
    f"LOCAL_WORLD_SIZE={os.environ['TASK_ENV_TP']}",
    f"CUDA_HOME={os.environ['CUDA_HOME']}",
    f"TRTLLM_DG_NVCC_COMPILER={os.environ['TRTLLM_DG_NVCC_COMPILER']}",
    f"PATH={os.environ['PATH']}",
    f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
]))
PY
)"

{
  echo "run start $(date '+%F %T')"
  cat "${RESULT_DIR}/manifest.txt"
  echo "env_json=${ENV_JSON}"
} > "${RESULT_DIR}/runner.log"

write_component_stats() {
  local output_file="$1"
  STAT_REMOTE="${REMOTE_JIT_DIR}" STAT_LOCAL="${LOCAL_JIT_DIR}" "${PYTHON_BIN}" - <<'PY' > "${output_file}"
import os
from pathlib import Path

roots = [
    ("remote", Path(os.environ["STAT_REMOTE"])),
    ("local",  Path(os.environ["STAT_LOCAL"])),
]
components = [
    "flashinfer", "deep_gemm", "tensorrt_llm_deep_gemm",
    "torch_extensions", "triton", "triton_autotune",
]
print("root_type\tcomponent\tpath\texists\tfiles\tdirs\tbytes")
for root_type, root in roots:
    for component in components:
        path = root / component
        files = dirs = total = 0
        if path.exists():
            for child in path.rglob("*"):
                try:
                    if child.is_dir():
                        dirs += 1
                    elif child.is_file():
                        files += 1
                        total += child.stat().st_size
                except OSError:
                    pass
        print(f"{root_type}\t{component}\t{path}\t{path.exists()}\t{files}\t{dirs}\t{total}")
PY
}

write_root_artifacts() {
  local output_file="$1"
  STAT_REMOTE="${REMOTE_JIT_DIR}" "${PYTHON_BIN}" - <<'PY' > "${output_file}"
import os
from pathlib import Path

root = Path(os.environ["STAT_REMOTE"])
patterns = [".jit_snapshot.tar.zst", ".delta"]
print("path\texists\tfiles\tdirs\tbytes")
for item in patterns:
    path = root / item
    files = dirs = total = 0
    if path.is_file():
        files = 1
        try:
            total = path.stat().st_size
        except OSError:
            total = 0
    elif path.is_dir():
        for child in path.rglob("*"):
            try:
                if child.is_dir():
                    dirs += 1
                elif child.is_file():
                    files += 1
                    total += child.stat().st_size
            except OSError:
                pass
    print(f"{path}\t{path.exists()}\t{files}\t{dirs}\t{total}")
PY
}

write_component_stats "${RESULT_DIR}/component_stats_before.tsv"
write_root_artifacts "${RESULT_DIR}/remote_artifacts_before.tsv"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, generated task info and manifest only." >> "${RESULT_DIR}/runner.log"
  echo "RESULT_DIR=${RESULT_DIR}"
  echo "TASK_INFO=${TASK_INFO}"
  echo "REMOTE_JIT_DIR=${REMOTE_JIT_DIR}"
  exit 0
fi

set +e
(
  cd "${REPO_ROOT}/rtp_llm/test/smoke"
  "${PYTHON_BIN}" -m smoke.entry \
    --suite_name jit_remote_cache_qwen35_moe_workloads \
    --task_info "${TASK_INFO}" \
    --envs "${ENV_JSON}" \
    --smoke_args "${SMOKE_ARGS}" \
    --gpu_card H20
) >> "${RESULT_DIR}/runner.log" 2>&1
rc=$?
set -e

{
  echo
  echo "run end $(date '+%F %T') rc=${rc}"
} >> "${RESULT_DIR}/runner.log"
echo "${rc}" > "${RESULT_DIR}/exit_code"

{
  echo "=== key log lines ==="
  rg "JIT cache|Backend manager start done|Backend server ready timing|Request performance|Request first token|completed=|ttft|e2e|latency|loaded JIT cache|remote restore|delta archive|compact|TensorRT-LLM|TRTLLM|nvcc|ERROR|Traceback" "${RESULT_DIR}" || true
} > "${RESULT_DIR}/key_metrics.log"

write_component_stats "${RESULT_DIR}/component_stats.tsv"
write_root_artifacts "${RESULT_DIR}/remote_artifacts.tsv"

RESULT_DIR_EXPORT="${RESULT_DIR}" TASK_INFO_EXPORT="${TASK_INFO}" "${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/request_metrics.tsv"
import json, os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR_EXPORT"])
task_info_path = Path(os.environ["TASK_INFO_EXPORT"])
workloads = []
try:
    task = json.loads(task_info_path.read_text())
    for item in task.get("query_result", []):
        workloads.append(item.get("_jit_workload", {}))
except Exception:
    workloads = []

rows = []
for path in sorted((result_dir / "main_logs").glob("access_r0_s*.log")):
    for line in path.read_text(errors="replace").splitlines():
        try:
            payload = json.loads(line)
            responses = payload["response"]["responses"]
            if not responses:
                continue
            aux = responses[0]["aux_info"]
        except Exception:
            continue
        rows.append({
            "log_time": payload.get("log_time", ""),
            "request_id": payload.get("id", ""),
            "input_len": aux.get("input_len", ""),
            "output_len": aux.get("output_len", ""),
            "cost_ms": aux.get("cost_time", ""),
            "ttft_ms": aux.get("first_token_cost_time", ""),
            "reuse_len": aux.get("reuse_len", ""),
        })
rows.sort(key=lambda r: r["log_time"])
for idx, row in enumerate(rows):
    wl = workloads[idx] if idx < len(workloads) else {}
    row["group"] = wl.get("group", "")
    row["workload"] = wl.get("name", "")
    row["prompt_words"] = wl.get("prompt_words", "")
    row["repeat_index"] = wl.get("repeat_index", "")
    row["max_new_tokens"] = wl.get("max_new_tokens", "")

cols = ["group","workload","prompt_words","repeat_index","max_new_tokens",
        "log_time","request_id","input_len","output_len","cost_ms","ttft_ms","reuse_len"]
print("\t".join(cols))
for row in rows:
    print("\t".join(str(row.get(c,"")) for c in cols))
PY

echo "RESULT_DIR=${RESULT_DIR}"
echo "REMOTE_JIT_DIR=${REMOTE_JIT_DIR}"
echo "exit_code=${rc}"
exit "${rc}"
