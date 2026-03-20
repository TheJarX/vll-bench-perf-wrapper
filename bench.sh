#!/bin/bash
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SRC_DIR/utils.sh"
set -e

# Adjust as needed for your GPU setup, remember to set CUDA_VISIBLE_DEVICES env var before running the script, e.g.: export CUDA_VISIBLE_DEVICES=0,1,2,3
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODELS=()

CONCURRENCIES=(1 16 32 64 128 256 512)

EXECUTION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./performance_results"
LOG_DIR="./performance_logs/serve_bench_${EXECUTION_TIMESTAMP}"


PORT=8888
HOST="127.0.0.1"
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=2048

SERVER_START_TIMEOUT=900
SERVER_POLL_INTERVAL=15

DATASET_NAME="sonnet"
DATASET_PATH="./sonnet.txt"
SONNET_INPUT_LEN=512
SONNET_OUTPUT_LEN=256
SONNET_PREFIX_LEN=200
NUM_PROMPTS=500
READY_CHECK_TIMEOUT_SEC=600
SEED=42
NUM_WARMUPS=50
NUM_RUNS=3
VLLM_EXTRA_ARGS=""
VERBOSE=0

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"


capture_args "$@"
ensure_models
cleanup_gpu
print_config
ensure_dataset

# echo $VLLM_EXTRA_ARGS
#  echo "vllm serve $MODEL --host $HOST --port $PORT --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-model-len $MAX_MODEL_LEN $VLLM_EXTRA_ARGS"
# exit 0

TOTAL_BENCHMARKS=$(( ${#MODELS[@]} * ${#CONCURRENCIES[@]} * NUM_RUNS ))
CURRENT_BENCH=0
for MODEL in "${MODELS[@]}"; do
    LABEL=$(model_to_label "$MODEL")

    for CONCURRENCY in "${CONCURRENCIES[@]}"; do
        for RUN in $(seq 1 "$NUM_RUNS"); do
            CURRENT_BENCH=$((CURRENT_BENCH + 1))
            RUN_LABEL="${LABEL}_c${CONCURRENCY}_run${RUN}"
            SERVER_LOG="${LOG_DIR}/${RUN_LABEL}_server.log"
            BENCH_LOG="${LOG_DIR}/${RUN_LABEL}_bench.log"

            echo "=============================================="
            echo "[${CURRENT_BENCH}/${TOTAL_BENCHMARKS}] ${RUN_LABEL}"
            echo "  Model:       $MODEL"
            echo "  Concurrency: $CONCURRENCY"
            echo "  Run:         $RUN of $NUM_RUNS"
            echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
            echo "=============================================="

            cleanup_gpu

            echo "[$RUN_LABEL] Starting vLLM server..."
            if [ "$VERBOSE" -eq 1 ]; then
                echo "[$RUN_LABEL] Running vLLM using the following command:"
             # TODO: improve this is done
                echo "vllm serve $MODEL --host $HOST --port $PORT --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-model-len $MAX_MODEL_LEN $VLLM_EXTRA_ARGS"
            fi
            vllm serve \
                "$MODEL" \
                --host "$HOST" \
                --port "$PORT" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --max-model-len "$MAX_MODEL_LEN" \
                --disable-log-requests \
                $VLLM_EXTRA_ARGS \
                > "$SERVER_LOG" 2>&1 &
            SERVER_PID=$!

            echo "[$RUN_LABEL] Waiting for server (max ${SERVER_START_TIMEOUT}s)..."
            elapsed=0
            while [ "$elapsed" -lt "$SERVER_START_TIMEOUT" ]; do
                if curl -s -o /dev/null --connect-timeout 5 "http://${HOST}:${PORT}/health" 2>/dev/null; then
                    echo "[$RUN_LABEL] Server ready after ${elapsed}s."
                    break
                fi
                sleep "$SERVER_POLL_INTERVAL"
                elapsed=$((elapsed + SERVER_POLL_INTERVAL))
            done
            if [ "$elapsed" -ge "$SERVER_START_TIMEOUT" ]; then
                echo "[$RUN_LABEL] ERROR: Server did not start. Check $SERVER_LOG"
                kill "$SERVER_PID" 2>/dev/null || true
                wait "$SERVER_PID" 2>/dev/null || true
                continue
            fi

            echo "[$RUN_LABEL] Running benchmark (concurrency=$CONCURRENCY)..."
            if vllm bench serve \
                --backend openai \
                --host "$HOST" \
                --port "$PORT" \
                --model "$MODEL" \
                --dataset-name "$DATASET_NAME" \
                --dataset-path "$DATASET_PATH" \
                --sonnet-input-len "$SONNET_INPUT_LEN" \
                --sonnet-output-len "$SONNET_OUTPUT_LEN" \
                --sonnet-prefix-len "$SONNET_PREFIX_LEN" \
                --num-prompts "$NUM_PROMPTS" \
                --max-concurrency "$CONCURRENCY" \
                --save-result \
                --result-dir "$RESULTS_DIR" \
                --label "${RUN_LABEL}" \
                --ready-check-timeout-sec "$READY_CHECK_TIMEOUT_SEC" \
                --seed "$SEED" \
                --num-warmups "$NUM_WARMUPS" \
                --temperature 0 \
                --metadata "gpus_used=$CUDA_VISIBLE_DEVICES" "model=$MODEL" "run=$RUN" "concurrency=$CONCURRENCY" "dataset=$DATASET_NAME" \
                2>&1 | tee -a "$BENCH_LOG"; then
                echo "[$RUN_LABEL] Benchmark completed."
            else
                echo "[$RUN_LABEL] Benchmark FAILED (see $BENCH_LOG)."
            fi

            echo "[$RUN_LABEL] Stopping server (PID $SERVER_PID)..."
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
            echo ""
        done
    done
done
echo "=============================================="
echo "All benchmarks complete. Generated results are in $RESULTS_DIR, logs are in $LOG_DIR"
echo "=============================================="

while true; do
    read -r -p "Use python script to generate results table? [y/N]: " response
    case "${response,,}" in
        y|yes)
            echo "Generating tables..."
            python "$SRC_DIR/process_results.py" "$RESULTS_DIR" || exit 1
        break ;;
        n|no)  echo "Cancelled."; exit 1 ;;
        *)     echo "No response, exiting"; exit 1 ;;
    esac
done