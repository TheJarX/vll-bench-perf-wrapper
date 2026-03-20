 #!/bin/bash

 capture_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --server-start-timeout)
            SERVER_START_TIMEOUT="$2"
            shift 2
            ;;
        --server-poll-interval)
            SERVER_POLL_INTERVAL="$2"
            shift 2
            ;;
        --ready-check-timeout-sec)
            READY_CHECK_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --concurrencies)
            IFS=',' read -ra CONCURRENCIES <<< "$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --sonnet-input-len)
            SONNET_INPUT_LEN="$2"
            shift 2
            ;;
        --sonnet-output-len)
            SONNET_OUTPUT_LEN="$2"
            shift 2
            ;;
        --sonnet-prefix-len)
            SONNET_PREFIX_LEN="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --warmups)
            NUM_WARMUPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --vllm-extra-args)
            # no escape handling here, just pass the string as is to the vLLM command. User is expected to provide a properly formatted string with necessary escaping if needed.
            # Ofc this is prone to user error but trying to do any better would require a much more complex parsing logic that goes beyond the scope of this script. e.g. if user wants to pass --tp 4 --trust-remote-code they should provide --vllm-extra-args "--tp 4 --trust-remote-code" and if they want to pass a single flag like --trust-remote-code they can provide --vllm-extra-args "--trust-remote-code". Also prone to issues if user provides flags that are already handled by the script, in which case the behaviour is undefined. Use at your own risk. Also the user could just override the whole command by providing something like --vllm-extra-args ";serve mymodel -tp 4 --trust-remote-code" in which case the behaviour is completely undefined and goes beyond the scope of this script, so again use at your own risk.
            VLLM_EXTRA_ARGS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            perf_help
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
    esac
  done
}
 perf_help() {
 echo "Usage: $0 --models /path/to/model [options]"
            echo "Options:"
            echo "  --models <model1,model2,...>          Comma-separated list of model paths to benchmark. Remember to provide the full path w/o the trailing slash. e.g. /path/to/model, not /path/to/model/"
            echo "  --results-dir <dir>                   Directory to save benchmark results (default: ./performance_results)"
            echo "  --log-dir <dir>                       Directory to save logs (default: ./performance_logs/serve_bench_TIMESTAMP)"
            echo "  --port <port>                         Port for vLLM server (default: 8888)"
            echo "  --host <host>                         Host for vLLM server (default:127.0.0.1)"
            echo "  --gpu-memory-utilization <float>      GPU memory utilization for vLLM server (default: 0.9)"
            echo "  --max-model-len <int>              Max model length for vLLM server (default: 2048)"
            echo "  --server-start-timeout <seconds>      Max time to wait for server to start (default: 900)"
            echo "  --server-poll-interval <seconds>     Interval to check if server is ready (default: 15)"
            echo "  --ready-check-timeout-sec <seconds>     Timeout for benchmark ready check (default: 600)"
            echo "  --concurrencies <c1,c2,...>           Comma-separated list of concurrencies to benchmark (default: 1,16,32,64,128,256,512)"
            echo "  --dataset-name <name>                 Name of the dataset (default: sonnet)"
            echo "  --dataset-path <path>                 Path to the dataset file (default: ./sonnet.txt)"
            echo "  --sonnet-input-len <int>             Input length for sonnet benchmark (default: 512)"
            echo "  --sonnet-output-len <int>            Output length for sonnet benchmark (default: 256)"
            echo "  --sonnet-prefix-len <int>            Prefix length for sonnet benchmark (default: 200)"
            echo "  --num-prompts <int>                  Number of prompts to use in the benchmark (default: 500)"
            echo "  --warmups <int>                     Number of warmup runs before benchmarking, set this to 0 to disable warmups. (default: 50)"
            echo "  --seed <int>                          Random seed for benchmarking (default: 42)"
            echo "  --num-runs <int>                       Number of runs per model/concurrency combo (default: 3)"
            echo "  --vllm-extra-args <args>              Extra arguments to pass to vLLM serve command (e.g. --vllm-extra-args \"-tp 4 --trust-remote-code\"). If a flag that is already handled is provided it'll probably override the existing one but it has no predictable behaviour and goes beyond the scope of this script. (default: '')"
            echo "  --verbose                             Enable verbose output (default: false)"
            echo "  --help                                Show this help message and exit"
            exit 0
 }

 model_to_label() {
    echo "$1" | sed 's|.*/||'

}

cleanup_gpu() {
    echo "Cleaning up GPU state..."

    pgrep -f "vllm serve.*--port $PORT" > /dev/null && echo "Warning: vLLM server processes detected on port $PORT. Attempting to kill..." || echo "No vLLM server processes detected on port $PORT."
    if pgrep -f "vllm serve.*--port $PORT" > /dev/null; then
        echo "Killing existing vLLM server processes on port $PORT..."
        pkill -f "vllm serve.*--port $PORT" 2>/dev/null || true
        sleep 5
    fi
    IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
    for gpu in "${gpus[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        dev="/dev/nvidia$gpu"
        if [ -e "$dev" ]; then
            local pids
            pids=$(lsof -t "$dev" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                echo "Killing processes using $dev: $pids"
                echo "$pids" | xargs kill -9 2>/dev/null || true
                sleep 3
            else
                echo "No processes found using $dev."
            fi
        fi
    done
    echo "Cleanup of processes using GPU done."
}

print_config() {
  echo "=============================================="
  echo "Performance Benchmark Configuration"
  echo "=============================================="
  echo "  Models:          ${MODELS[*]}"
  echo "  Concurrencies:   ${CONCURRENCIES[*]}"
  echo "  Dataset:         ${DATASET_NAME} (${DATASET_PATH})"
  echo "  Sonnet input:    ${SONNET_INPUT_LEN} tokens"
  echo "  Sonnet output:   ${SONNET_OUTPUT_LEN} tokens"
  echo "  Sonnet prefix:   ${SONNET_PREFIX_LEN} tokens"
  echo "  Num prompts:     ${NUM_PROMPTS}"
  echo "  Warmups:         ${NUM_WARMUPS}"
  echo "  Seed:            ${SEED}"
  echo "  Runs per combo:  ${NUM_RUNS}"
  echo "  Results dir:     ${RESULTS_DIR}"
  echo "  Log dir:         ${LOG_DIR}"
  echo "=============================================="
  echo ""
}

ensure_dataset() {
    if [ ! -f "$DATASET_PATH" ]; then
        echo -e "\033[33mDownloading sonnet.txt...\033[0m";
        curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt" -o "$DATASET_PATH"
    else
        echo "Dataset already exists at $DATASET_PATH, skipping download."
    fi
}

ensure_models() {
    if [ ${#MODELS[@]} -eq 0 ]; then
        err_n_ex "No models specified. Use \033[0;41m--models\033[0;31m to specify at least one model."
        exit 1
    fi
}

err_n_ex() {
    echo -e "\033[31mERROR: $1\033[0m"
    echo "Use --help to see usage."
    exit 1
}