#!/bin/bash
# ============================================================
# vLLM 模型启动脚本(后台运行版)
# 用法:
#   bash start_model.sh                      交互式选择模型启动
#   bash start_model.sh <模型目录名>          直接指定模型启动
#   bash start_model.sh <模型目录名> <端口> <最大上下文>
#
# 预设模型(未下载时会自动从 HuggingFace 拉取):
#   1. Qwen3.6-35B-A3B
#   2. Llama-3.3-70B  (FP8, 适配 RTX PRO 6000 96GB 单卡)
# ============================================================

STATE_FILE="/workspace/.vllm_service"
LOG_DIR="/workspace/logs"
MODELS_DIR="/workspace/models"
mkdir -p "$LOG_DIR" "$MODELS_DIR"

# 预设模型: 本地目录名 | HuggingFace 仓库 | reasoning-parser(空=不使用) | 默认最大上下文
PRESET_NAMES=(
    "Qwen3.6-35B-A3B"
    "Llama-3.3-70B"
)
PRESET_REPOS=(
    "Qwen/Qwen3.6-35B-A3B"
    "nvidia/Llama-3.3-70B-Instruct-FP8"
)
PRESET_REASONING=(
    "qwen3"
    ""
)
PRESET_MAX_LEN=(
    131072
    131072
)

MODEL_NAME=$1
PORT=${2:-8000}
MAX_LEN=$3
REASONING_PARSER=""
VLLM_MODEL_ARGS=()

# ------------------------------------------------------------
# 预设模型辅助函数
# ------------------------------------------------------------
is_preset_model() {
    local name=$1
    for preset in "${PRESET_NAMES[@]}"; do
        [ "$preset" = "$name" ] && return 0
    done
    return 1
}

get_preset_index() {
    local name=$1
    for i in "${!PRESET_NAMES[@]}"; do
        [ "${PRESET_NAMES[$i]}" = "$name" ] && echo "$i" && return 0
    done
    return 1
}

model_is_ready() {
    local dir=$1
    [ -d "$dir" ] && [ -f "$dir/config.json" ]
}

ensure_hf_cli() {
    if [ -d "/workspace/venv" ]; then
        source /workspace/venv/bin/activate
    fi
    if ! command -v hf &> /dev/null; then
        echo "[提示] 未检测到 hf CLI,正在安装 huggingface_hub..."
        pip install -U "huggingface_hub[cli]"
    fi
}

download_model() {
    local name=$1
    local repo=$2
    local target="${MODELS_DIR}/${name}"

    ensure_hf_cli

    echo "=========================================="
    echo "  模型未下载,开始自动下载"
    echo "  名称: $name"
    echo "  仓库: $repo"
    echo "  目标: $target"
    echo "=========================================="

    if [ -z "${HF_TOKEN:-}" ] && [[ "$repo" == meta-llama/* ]]; then
        echo "[提示] Meta Llama 模型需要 HuggingFace 授权,请先设置 HF_TOKEN:"
        echo "  export HF_TOKEN=你的token"
        echo "  并在 https://huggingface.co/$repo 同意许可协议。"
    fi

    mkdir -p "$target"
    if ! hf download "$repo" --local-dir "$target"; then
        echo "[错误] 模型下载失败: $repo"
        exit 1
    fi

    if ! model_is_ready "$target"; then
        echo "[错误] 下载完成但未找到有效的 config.json: $target"
        exit 1
    fi

    echo "模型下载完成: $target"
}

ensure_model_available() {
    local name=$1
    local dir="${MODELS_DIR}/${name}"

    if model_is_ready "$dir"; then
        return 0
    fi

    if is_preset_model "$name"; then
        local idx
        idx=$(get_preset_index "$name")
        download_model "$name" "${PRESET_REPOS[$idx]}"
        return 0
    fi

    echo "[错误] 找不到模型目录: $dir"
    echo "       可用预设模型: ${PRESET_NAMES[*]}"
    exit 1
}

apply_preset_launch_config() {
    local name=$1
    if ! is_preset_model "$name"; then
        return 0
    fi

    local idx
    idx=$(get_preset_index "$name")
    REASONING_PARSER="${PRESET_REASONING[$idx]}"
    if [ -z "$MAX_LEN" ]; then
        MAX_LEN="${PRESET_MAX_LEN[$idx]}"
    fi

    VLLM_MODEL_ARGS=()
    case "$name" in
        Llama-3.3-70B)
            # RTX PRO 6000 (96GB): FP8 权重约 70GB,单卡可跑 128K 上下文
            VLLM_MODEL_ARGS=(
                --kv-cache-dtype fp8
                --async-scheduling
                --gpu-memory-utilization 0.92
            )
            ;;
    esac
}

# ------------------------------------------------------------
# 1. 检测是否已有服务在运行
# ------------------------------------------------------------
is_running() {
    if [ -f "$STATE_FILE" ]; then
        RUNNING_PID=$(grep "^PID=" "$STATE_FILE" | cut -d= -f2)
        if [ -n "$RUNNING_PID" ] && kill -0 "$RUNNING_PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

if is_running; then
    source "$STATE_FILE"
    echo "=========================================="
    echo "  检测到已有模型服务正在运行"
    echo "=========================================="
    echo "  模型: $RUNNING_MODEL"
    echo "  端口: $RUNNING_PORT"
    echo "  PID:  $RUNNING_PID"
    echo "  日志: $RUNNING_LOG"
    echo "=========================================="
    read -p "是否停止当前服务? (y/n): " STOP_IT
    if [ "$STOP_IT" = "y" ] || [ "$STOP_IT" = "Y" ]; then
        kill "$RUNNING_PID" 2>/dev/null
        sleep 2
        kill -0 "$RUNNING_PID" 2>/dev/null && kill -9 "$RUNNING_PID" 2>/dev/null
        rm -f "$STATE_FILE"
        echo "已停止。"
    else
        echo "保留当前运行的服务,退出脚本。"
        exit 0
    fi
fi

# ------------------------------------------------------------
# 2. 确定要启动的模型(未指定则交互式选择)
# ------------------------------------------------------------
if [ -z "$MODEL_NAME" ]; then
    echo "请选择要启动的模型:"
    for i in "${!PRESET_NAMES[@]}"; do
        name="${PRESET_NAMES[$i]}"
        if model_is_ready "${MODELS_DIR}/${name}"; then
            status="已下载"
        else
            status="未下载(将自动下载)"
        fi
        echo "  $((i+1))) ${name}  [$status]"
    done

    EXTRA_MODELS=()
    for existing in "$MODELS_DIR"/*; do
        [ -d "$existing" ] || continue
        base=$(basename "$existing")
        is_preset_model "$base" && continue
        model_is_ready "$existing" || continue
        EXTRA_MODELS+=("$base")
    done

    MENU_OFFSET=${#PRESET_NAMES[@]}
    for i in "${!EXTRA_MODELS[@]}"; do
        echo "  $((MENU_OFFSET + i + 1))) ${EXTRA_MODELS[$i]}  [已下载]"
    done

    read -p "输入序号: " CHOICE
    if ! [[ "$CHOICE" =~ ^[0-9]+$ ]]; then
        echo "[错误] 无效选择。"
        exit 1
    fi

    INDEX=$((CHOICE - 1))
    if [ "$INDEX" -ge 0 ] && [ "$INDEX" -lt "${#PRESET_NAMES[@]}" ]; then
        MODEL_NAME="${PRESET_NAMES[$INDEX]}"
    elif [ "$INDEX" -ge "$MENU_OFFSET" ] && [ "$INDEX" -lt "$((MENU_OFFSET + ${#EXTRA_MODELS[@]}))" ]; then
        MODEL_NAME="${EXTRA_MODELS[$((INDEX - MENU_OFFSET))]}"
    else
        echo "[错误] 无效选择。"
        exit 1
    fi
fi

apply_preset_launch_config "$MODEL_NAME"
MAX_LEN=${MAX_LEN:-131072}
ensure_model_available "$MODEL_NAME"

MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"
if ! model_is_ready "$MODEL_DIR"; then
    echo "[错误] 模型目录无效: $MODEL_DIR"
    exit 1
fi

# ------------------------------------------------------------
# 3. 环境准备
# ------------------------------------------------------------
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"

if [ -d "/workspace/venv" ]; then
    source /workspace/venv/bin/activate
fi

LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "  启动模型: $MODEL_NAME"
if [ "$MODEL_NAME" = "Llama-3.3-70B" ]; then
    echo "  精度: FP8 (nvidia/Llama-3.3-70B-Instruct-FP8)"
fi
echo "  端口: $PORT"
echo "  最大上下文长度: $MAX_LEN"
echo "  日志文件: $LOG_FILE"
echo "=========================================="

cd "$MODEL_DIR" || exit 1

# ------------------------------------------------------------
# 4. 后台启动(nohup + disown,关闭终端也不会中断)
# ------------------------------------------------------------
# 以下参数针对 Blackwell 架构显卡(如 RTX PRO 6000,SM120)的兼容性问题,
# 如果显卡非 Blackwell 或 CUDA >= 12.9,可去掉这几个参数以获得更好性能:
#   --moe-backend triton --attention-backend triton_attn
#   VLLM_USE_FLASHINFER_SAMPLER=0

VLLM_EXTRA_ARGS=()
if [ -n "$REASONING_PARSER" ]; then
    VLLM_EXTRA_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

nohup env VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "." \
  --served-model-name "${MODEL_NAME}" \
  --max-model-len "$MAX_LEN" \
  "${VLLM_EXTRA_ARGS[@]}" \
  "${VLLM_MODEL_ARGS[@]}" \
  --port "$PORT" \
  --moe-backend triton \
  --attention-backend triton_attn \
  --max-num-seqs 512 \
  > "$LOG_FILE" 2>&1 &

NEW_PID=$!
disown

# 保存运行状态
cat > "$STATE_FILE" << EOF
PID=$NEW_PID
RUNNING_PID=$NEW_PID
RUNNING_MODEL=$MODEL_NAME
RUNNING_PORT=$PORT
RUNNING_LOG=$LOG_FILE
EOF

echo "已在后台启动,PID: $NEW_PID"
echo "实时查看完整日志: tail -f $LOG_FILE"
echo ""

# ------------------------------------------------------------
# 5. 等待服务就绪(逐阶段输出进度,而非静默等待)
# ------------------------------------------------------------
# 每个阶段对应日志里会出现的关键字,按顺序检测,检测到就打印一次
declare -a STAGE_KEYWORDS=(
    "Starting to load model|开始加载模型权重"
    "Loading weights took|模型权重加载完成"
    "gpu_model_runner.py:5255|模型已载入显存"
    "Dynamo bytecode transform|开始图编译(torch.compile)"
    "torch.compile took|图编译完成"
    "Initial profiling/warmup run took|显存/性能 profiling 完成"
    "GPU KV cache size|KV Cache 分配完成"
    "Uvicorn running|API 服务已就绪"
)

declare -A STAGE_DONE
TOTAL_STAGES=${#STAGE_KEYWORDS[@]}
DONE_COUNT=0
READY=0
MAX_WAIT=300   # 最多等待 5 分钟
ELAPSED=0

echo "----- 启动进度 -----"

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$NEW_PID" 2>/dev/null; then
        echo ""
        echo "[错误] 进程已退出,启动失败,最后 30 行日志如下:"
        echo "--------------------------------------------"
        tail -n 30 "$LOG_FILE"
        echo "--------------------------------------------"
        rm -f "$STATE_FILE"
        exit 1
    fi

    for idx in "${!STAGE_KEYWORDS[@]}"; do
        IFS='|' read -r PATTERN LABEL <<< "${STAGE_KEYWORDS[$idx]}"
        if [ -z "${STAGE_DONE[$idx]}" ] && grep -qE "$PATTERN" "$LOG_FILE" 2>/dev/null; then
            STAGE_DONE[$idx]=1
            DONE_COUNT=$((DONE_COUNT+1))
            echo "  [$DONE_COUNT/$TOTAL_STAGES] ✔ $LABEL"
            if [ "$PATTERN" = "Uvicorn running" ]; then
                READY=1
            fi
        fi
    done

    [ "$READY" -eq 1 ] && break

    sleep 2
    ELAPSED=$((ELAPSED+2))
done

echo "---------------------"
echo ""

if [ "$READY" -eq 1 ]; then
    echo "服务已就绪!(总耗时约 ${ELAPSED} 秒)"
else
    echo "[提示] 等待超过 5 分钟仍未完全就绪,进程仍在运行,可能是模型较大编译较慢。"
    echo "       继续用以下命令查看实时日志: tail -f $LOG_FILE"
fi

echo ""
echo "=========================================="
echo "  测试命令"
echo "=========================================="
echo ""
echo "curl http://localhost:${PORT}/v1/chat/completions \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"你好,简单介绍一下你自己\"}], \"max_tokens\": 200}' \\"
echo "  | python3 -m json.tool"
echo ""
echo "下次运行本脚本可查看/停止当前服务,或切换启动其他模型。"
