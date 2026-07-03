#!/bin/bash
# ============================================================
# Runpod 环境检测 + 自动安装配置脚本
# 用法: source setup_env.sh   (务必用 source,不要用 bash 执行,
#       否则虚拟环境激活状态无法保留在当前终端)
# ============================================================

echo "=========================================="
echo "  1. GPU / CUDA 环境检测"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
else
    echo "[警告] nvidia-smi 未找到,GPU 可能未正确挂载"
fi

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "CUDA 编译工具链版本: $NVCC_VERSION"
    # 判断是否满足 Blackwell(SM120)架构所需的 12.9+
    MAJOR=$(echo "$NVCC_VERSION" | cut -d. -f1)
    MINOR=$(echo "$NVCC_VERSION" | cut -d. -f2)
    if [ "$MAJOR" -lt 12 ] || { [ "$MAJOR" -eq 12 ] && [ "$MINOR" -lt 9 ]; }; then
        echo "[提示] CUDA < 12.9,如果显卡是 Blackwell 架构(如 RTX PRO 6000/RTX 50系列),"
        echo "       启动 vLLM 时可能需要加 --moe-backend triton --attention-backend triton_attn"
        echo "       以及 VLLM_USE_FLASHINFER_SAMPLER=0 绕开 FlashInfer 兼容性问题。"
    fi
else
    echo "[警告] nvcc 未找到"
fi

echo ""
echo "=========================================="
echo "  2. SSH 服务检测与修复"
echo "=========================================="

if service ssh status &> /dev/null; then
    echo "sshd 已在运行"
else
    echo "sshd 未运行,尝试修复..."
    ssh-keygen -A
    service ssh start
    if service ssh status &> /dev/null; then
        echo "sshd 修复成功"
    else
        echo "[警告] sshd 仍未能启动,请手动排查"
    fi
fi

echo ""
echo "=========================================="
echo "  2.1 authorized_keys(登录公钥)检测"
echo "=========================================="

mkdir -p ~/.ssh
chmod 700 ~/.ssh
AUTH_KEYS=~/.ssh/authorized_keys

if [ -s "$AUTH_KEYS" ]; then
    echo "已检测到 authorized_keys,当前已授权的公钥数量: $(wc -l < "$AUTH_KEYS")"
else
    echo "[提示] 未检测到 authorized_keys,或文件为空。"
    echo "如果你之后用 SSH 直连时提示 Permission denied / 要求输入密码,通常就是这里没有公钥。"
    read -p "是否现在粘贴你的公钥内容并保存? (y/n): " ADD_KEY
    if [ "$ADD_KEY" = "y" ] || [ "$ADD_KEY" = "Y" ]; then
        echo "请粘贴完整公钥内容(以 ssh-ed25519 或 ssh-rsa 开头的那一整行),粘贴后按回车:"
        read -r PUB_KEY
        if [ -n "$PUB_KEY" ]; then
            echo "$PUB_KEY" >> "$AUTH_KEYS"
            chmod 600 "$AUTH_KEYS"
            echo "已保存,当前 authorized_keys 内容:"
            cat "$AUTH_KEYS"
        else
            echo "输入为空,已跳过。"
        fi
    else
        echo "已跳过,如需要可稍后手动执行:"
        echo "  echo \"你的公钥\" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    fi
fi

echo ""
echo "=========================================="
echo "  3. Network Volume 挂载检测"
echo "=========================================="

if [ -d "/workspace" ]; then
    df -h /workspace
else
    echo "[警告] /workspace 目录不存在,请检查 Network Volume 是否正确挂载"
fi

echo ""
echo "=========================================="
echo "  4. Python 虚拟环境(/workspace/venv)"
echo "=========================================="

if [ -d "/workspace/venv" ]; then
    echo "检测到已有虚拟环境,直接激活"
    source /workspace/venv/bin/activate
else
    echo "未检测到虚拟环境,正在创建..."
    python3 -m venv /workspace/venv
    source /workspace/venv/bin/activate
    pip install --upgrade pip
fi

echo "当前 Python: $(which python3)"

echo ""
echo "=========================================="
echo "  5. 依赖库检测与安装"
echo "=========================================="

if python3 -c "import vllm" &> /dev/null; then
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
    echo "vllm 已安装,版本: $VLLM_VER"
else
    echo "vllm 未安装,正在安装..."
    pip install vllm
fi

if command -v hf &> /dev/null; then
    echo "huggingface_hub CLI 已安装: $(hf --version)"
else
    echo "huggingface_hub[cli] 未安装,正在安装..."
    pip install -U "huggingface_hub[cli]"
fi

echo ""
echo "=========================================="
echo "  6. 编译缓存目录设置(持久化到 /workspace)"
echo "=========================================="

export VLLM_CACHE_ROOT=/workspace/.cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"
echo "VLLM_CACHE_ROOT 已设置为: $VLLM_CACHE_ROOT"
echo "(建议把这行加入 ~/.bashrc,或每次 source 本脚本时自动生效)"

echo ""
echo "=========================================="
echo "  7. 模型目录检测"
echo "=========================================="

mkdir -p /workspace/models
echo "已存在的模型:"
ls -1 /workspace/models 2>/dev/null || echo "  (空,尚未下载任何模型)"

echo ""
echo "=========================================="
echo "  环境检测与配置完成"
echo "=========================================="
