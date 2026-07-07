#!/usr/bin/env bash
set -euo pipefail

# Start one DeepSeek-V3.2-4layer smoke server and send configurable prompt
# lengths to force different JIT kernel shapes. Each length is sent at least
# twice so the second request can show same-shape cache reuse in logs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" && -x "/usr/local/cuda-12.9/bin/nvcc" ]]; then
  CUDA_HOME="/usr/local/cuda-12.9"
fi
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"

MODEL_TYPE="${MODEL_TYPE:-deepseek_v32}"
MODEL_PATH="${MODEL_PATH:-/mnt/nas1/hf/DeepSeek-V3.2-4layer}"
GPU="${GPU:-7}"

REMOTE_JIT_DIR="${REMOTE_JIT_DIR:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/data1/ganyiwei.gyw/RTP-LLM/jit_cache_results/${RUN_ID}}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/deepseek_v32_jit_workloads}"
LOCAL_JIT_DIR="${LOCAL_JIT_DIR:-${RESULT_DIR}/local_jit}"

# Prompt lengths to send. PROMPT_WORDS_LIST overrides SHORT_WORDS/LONG_WORDS
# and should be a comma separated list, for example: 64,128,256,512.
SHORT_WORDS="${SHORT_WORDS:-64}"
LONG_WORDS="${LONG_WORDS:-256}"
PROMPT_WORDS_LIST="${PROMPT_WORDS_LIST:-${SHORT_WORDS},${LONG_WORDS}}"
REPEAT_PER_LENGTH="${REPEAT_PER_LENGTH:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"

SMOKE_ARGS="${SMOKE_ARGS:---warm_up 0 --seq_size_per_block 64 --act_type BF16 --reserver_runtime_mem_mb 49343 --enable_cuda_graph 0 --reuse_cache 1 --hack_layer_num 1 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1 --load_method scratch}"

if [[ -z "${REMOTE_JIT_DIR}" ]]; then
  echo "REMOTE_JIT_DIR is required, for example:" >&2
  echo "  REMOTE_JIT_DIR=/mnt/nas1/rtp_llm_jit_cache/nonsnapshot/${RUN_ID} $0" >&2
  exit 2
fi
if [[ ! -x "${NVCC_BIN}" ]]; then
  echo "nvcc is required for TensorRT-LLM DeepGEMM JIT but was not found: ${NVCC_BIN}" >&2
  exit 2
fi

mkdir -p "${REMOTE_JIT_DIR}" "${LOCAL_JIT_DIR}" "${RESULT_DIR}"

PROTO_DIR="${REPO_ROOT}/rtp_llm/cpp/model_rpc/proto"
PROTO_GEN_DIR="${PROTO_GEN_DIR:-/tmp/rtp_proto_gen_check/rtp_llm/cpp/model_rpc/proto}"
if [[ ! -e "${PROTO_DIR}/model_rpc_service_pb2.py" && -e "${PROTO_GEN_DIR}/model_rpc_service_pb2.py" ]]; then
  ln -s "${PROTO_GEN_DIR}/model_rpc_service_pb2.py" "${PROTO_DIR}/model_rpc_service_pb2.py"
fi
if [[ ! -e "${PROTO_DIR}/model_rpc_service_pb2_grpc.py" && -e "${PROTO_GEN_DIR}/model_rpc_service_pb2_grpc.py" ]]; then
  ln -s "${PROTO_GEN_DIR}/model_rpc_service_pb2_grpc.py" "${PROTO_DIR}/model_rpc_service_pb2_grpc.py"
fi

TASK_INFO="${RESULT_DIR}/deepseek_v32_jit_workloads.json"
export TASK_INFO MODEL_TYPE MODEL_PATH PROMPT_WORDS_LIST REPEAT_PER_LENGTH MAX_NEW_TOKENS
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

prompt_words_list = [
    int(item.strip())
    for item in os.environ["PROMPT_WORDS_LIST"].split(",")
    if item.strip()
]
if not prompt_words_list:
    raise ValueError("PROMPT_WORDS_LIST must contain at least one length")
repeat = int(os.environ["REPEAT_PER_LENGTH"])
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])

def prompt(words: int) -> str:
    return " ".join(["hello"] * words)

query_result = []
for words in prompt_words_list:
    for i in range(repeat):
        query_result.append(
            {
                "query": {
                    "prompt": [prompt(words)],
                    "generate_config": {
                        "max_new_tokens": max_new_tokens,
                        "top_k": 1,
                        "top_p": 0,
                    },
                },
                "result": {"response": "", "finished": True},
                "_jit_workload": {
                    "name": f"len_{words}",
                    "repeat_index": i + 1,
                    "prompt_words": words,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )

task = {
    "model_type": os.environ["MODEL_TYPE"],
    "model_path": os.environ["MODEL_PATH"],
    "query_result": query_result,
}
Path(os.environ["TASK_INFO"]).write_text(
    json.dumps(task, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

cat > "${RESULT_DIR}/manifest.txt" <<EOF
run_id=${RUN_ID}
repo_root=${REPO_ROOT}
model_type=${MODEL_TYPE}
model_path=${MODEL_PATH}
gpu=${GPU}
remote_jit_dir=${REMOTE_JIT_DIR}
local_jit_dir=${LOCAL_JIT_DIR}
cuda_home=${CUDA_HOME}
nvcc_bin=${NVCC_BIN}
result_dir=${RESULT_DIR}
task_info=${TASK_INFO}
short_words=${SHORT_WORDS}
long_words=${LONG_WORDS}
prompt_words_list=${PROMPT_WORDS_LIST}
repeat_per_length=${REPEAT_PER_LENGTH}
max_new_tokens=${MAX_NEW_TOKENS}
workloads=${PROMPT_WORDS_LIST} x ${REPEAT_PER_LENGTH}
smoke_args=${SMOKE_ARGS}
EOF

export TEST_UNDECLARED_OUTPUTS_DIR="${RESULT_DIR}"
export REMOTE_JIT_DIR
export LOCAL_JIT_DIR
export RESULT_DIR
export JIT_CACHE_DETAILED_STATS=1
export SAVE_RESPONSE=True
export ACCL_LOW_LATENCY_OPTIMIZE=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME
export TRTLLM_DG_NVCC_COMPILER="${TRTLLM_DG_NVCC_COMPILER:-${NVCC_BIN}}"
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/rtp_llm/test:${PYTHONPATH:-}"

ENV_JSON="$(TASK_ENV_REMOTE="${REMOTE_JIT_DIR}" TASK_ENV_LOCAL="${LOCAL_JIT_DIR}" TASK_ENV_GPU="${GPU}" "${PYTHON_BIN}" - <<'PY'
import json
import os

print(json.dumps([
    f"REMOTE_JIT_DIR={os.environ['TASK_ENV_REMOTE']}",
    f"LOCAL_JIT_DIR={os.environ['TASK_ENV_LOCAL']}",
    "JIT_CACHE_DETAILED_STATS=1",
    "SAVE_RESPONSE=True",
    "ACCL_LOW_LATENCY_OPTIMIZE=1",
    f"CUDA_VISIBLE_DEVICES={os.environ['TASK_ENV_GPU']}",
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
  "${PYTHON_BIN}" - <<'PY' > "${output_file}"
import os
from pathlib import Path

roots = [
    ("remote", Path(os.environ["REMOTE_JIT_DIR"])),
    ("local", Path(os.environ["LOCAL_JIT_DIR"])),
]
components = [
    "flashinfer",
    "deep_gemm",
    "tensorrt_llm_deep_gemm",
    "torch_extensions",
    "triton",
    "triton_autotune",
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
  "${PYTHON_BIN}" - <<'PY' > "${output_file}"
import os
from pathlib import Path

root = Path(os.environ["REMOTE_JIT_DIR"])
patterns = [
    ".jit_snapshot.tar.zst",
    ".delta",
]
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
    --suite_name jit_remote_cache_deepseek_v32_workloads \
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

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/request_metrics.tsv"
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
task_info = Path(os.environ["TASK_INFO"])
workloads = []
try:
    task = json.loads(task_info.read_text())
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
        rows.append(
            {
                "log_time": payload.get("log_time", ""),
                "request_id": payload.get("id", ""),
                "input_len": aux.get("input_len", ""),
                "output_len": aux.get("output_len", ""),
                "cost_ms": aux.get("cost_time", ""),
                "ttft_ms": aux.get("first_token_cost_time", ""),
                "reuse_len": aux.get("reuse_len", ""),
                "local_reuse_len": aux.get("local_reuse_len", ""),
                "remote_reuse_len": aux.get("remote_reuse_len", ""),
            }
        )
rows.sort(key=lambda item: item["log_time"])
for index, row in enumerate(rows):
    workload = workloads[index] if index < len(workloads) else {}
    row["workload"] = workload.get("name", "")
    row["prompt_words"] = workload.get("prompt_words", "")
    row["repeat_index"] = workload.get("repeat_index", "")
    row["max_new_tokens"] = workload.get("max_new_tokens", "")
print(
    "workload\tprompt_words\trepeat_index\tmax_new_tokens\tlog_time\trequest_id\tinput_len\toutput_len\tcost_ms\tttft_ms\treuse_len\tlocal_reuse_len\tremote_reuse_len"
)
for row in rows:
    print(
        "\t".join(
            str(row[key])
            for key in (
                "workload",
                "prompt_words",
                "repeat_index",
                "max_new_tokens",
                "log_time",
                "request_id",
                "input_len",
                "output_len",
                "cost_ms",
                "ttft_ms",
                "reuse_len",
                "local_reuse_len",
                "remote_reuse_len",
            )
        )
    )
PY

echo "RESULT_DIR=${RESULT_DIR}"
echo "REMOTE_JIT_DIR=${REMOTE_JIT_DIR}"
echo "exit_code=${rc}"
exit "${rc}"
