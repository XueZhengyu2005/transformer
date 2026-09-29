#!/bin/bash
# 模型下载脚本（使用国内镜像，缓存到大磁盘）
# 用法: bash download_models.sh          # 只下载核心小模型（约4GB）
#       bash download_models.sh --all    # 包含大模型 ChatGLM-6B / OpenLLaMA（约30GB）
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

CORE_MODELS=(
    "bert-base-chinese"                                # 文本分类 / PET / P-Tuning（约400MB）
    "nghuyong/ernie-3.0-base-zh"                       # 文本匹配x4 / RLHF奖励模型（约400MB）
    "Pky/uie-base-zh"                                  # UIE 信息抽取（约1.1GB）
    "uer/t5-base-chinese-cluecorpussmall"              # 问答生成 / 填空数据增强（约1.1GB）
    "uer/gpt2-chinese-cluecorpussmall"                 # RLHF PPO 语言模型（约500MB）
    "uer/roberta-base-finetuned-jd-binary-chinese"     # RLHF 情感打分模型（约400MB）
)

LARGE_MODELS=(
    "THUDM/chatglm-6b"                                 # LLM zero-shot / ChatGLM微调（约13GB）
    "openlm-research/open_llama_7b_v2"                 # LLMsTrainer（约14GB，可选）
)

echo "==== 下载核心模型（6个，约4GB）===="
for model in "${CORE_MODELS[@]}"; do
    echo ">>> $model"
    huggingface-cli download "$model"
done

if [ "$1" == "--all" ]; then
    echo "==== 下载大模型（2个，约27GB）===="
    for model in "${LARGE_MODELS[@]}"; do
        echo ">>> $model"
        huggingface-cli download "$model"
    done
fi

echo "==== 全部下载完成，缓存在: $HF_HOME ===="
df -h /root/autodl-tmp | tail -1
