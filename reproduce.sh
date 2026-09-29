#!/bin/bash
# ============================================================
# transformers_tasks 全项目复现脚本（可断点续跑）
# 用法: bash reproduce.sh
# 说明: 大数据集(SimCSE/RLHF/填空模型)已做抽样以保证总时长可控
#       已完成的任务（存在 checkpoints/*/model_best）会自动跳过
# ============================================================
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_ROOT="${MODEL_ROOT:-$HOME/autodl-tmp/upload_models}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/trans/bin/python}"
export BASE_PYTHON_BIN="${BASE_PYTHON_BIN:-$HOME/miniconda3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN=python; fi
if [ ! -x "$BASE_PYTHON_BIN" ]; then BASE_PYTHON_BIN=python; fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
if [ "${1:-}" = "--smoke" ]; then
    echo "一次性冒烟测试已清理。请按根目录 readme.md 从各任务目录运行 inference.py。" >&2
    exit 2
fi
RESULTS_DIR="reproduce_results"
mkdir -p "$RESULTS_DIR"
STATUS_FILE="$RESULTS_DIR/status.txt"
: > "$STATUS_FILE"
RUN_FAILURES=0

run_task() {
    local name="$1"; local done_marker="$2"; shift 2
    if [ -f "$done_marker" ]; then
        # A saved T5 config does not certify answer quality.  Their separate
        # evaluation must pass before the project can report all tasks done;
        # do not silently rerun the legacy training here or overwrite weights.
        if [[ "$name" == 11_* ]]; then
            if "$PYTHON_BIN" tools/check_qa_acceptance.py; then
                echo "PASS $name verified README QA example" >> "$STATUS_FILE"
                echo ">>> [已复核] $name 的 checkpoint 与完整 dev 报告一致；README 样例正确"
            else
                echo "FAIL $name QA validation" >> "$STATUS_FILE"
                echo ">>> [待验收] $name 已有权重，但独立验证未通过"
                RUN_FAILURES=1
            fi
            return 0
        elif [[ "$name" == 12_* ]]; then
            if "$BASE_PYTHON_BIN" tools/check_filling_basic.py; then
                echo "PASS $name README filling demo; see experiment report" >> "$STATUS_FILE"
                echo ">>> [基本演示通过] $name 可补全 README 示例；泛化表现见实验报告"
            else
                echo "FAIL $name filling demonstration" >> "$STATUS_FILE"
                echo ">>> [待验收] $name 的示例或模型产物无法核对"
                RUN_FAILURES=1
            fi
            return 0
        fi
        # A historical marker is not sufficient evidence for PPO or the
        # ChatGLM demos.  PPO must have a final model; the demos must be
        # rerun through their semantic validators.
        if [[ "$name" == "10_rlhf_ppo" && ! -f "RLHF/checkpoints/ppo_sentiment_gpt/final/config.json" ]]; then
            echo ">>> [重试] $name 的完成标记没有对应 final checkpoint"
        elif [[ "$name" == 13_* || "$name" == 14_* || "$name" == 15_* ]]; then
            echo ">>> [重试] $name 使用语义校验重新验证"
        else
            echo "SKIP $name artifact present; run its README inference for function verification" >> "$STATUS_FILE"
            echo ">>> [跳过] $name 有产物（存在 $done_marker），功能请按该任务 README 运行推理脚本核验"
            return 0
        fi
    fi
    echo ""
    echo "============================================================"
    echo ">>> [$(date '+%H:%M:%S')] 开始任务: $name"
    echo "============================================================"
    local START=$(date +%s)
    set +e
    "$@" 2>&1 | tee "$RESULTS_DIR/${name}.log" | grep -E "global step|Evaluation|best|F1|f1|acc|loss:|reward|Done|=====" | tail -50
    local status=${PIPESTATUS[0]}
    local END=$(date +%s)
    if [ "$status" -eq 0 ]; then
        echo "PASS $name $((END-START))s" >> "$STATUS_FILE"
        echo ">>> 任务 $name 成功，耗时 $((END-START)) 秒"
    else
        RUN_FAILURES=1
        echo "FAIL $name $((END-START))s" >> "$STATUS_FILE"
        echo ">>> 任务 $name 失败(exit=$status)，继续后续任务，详见 $RESULTS_DIR/${name}.log"
    fi
}

cd /root/autodl-tmp/transformers_tasks

# 1. 文本分类
run_task "01_text_classification" "text_classification/checkpoints/comment_classify/model_best/config.json" \
    bash -c 'cd text_classification && $PYTHON_BIN train.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify" \
    --img_log_dir "logs/comment_classify" --img_log_name "BERT" \
    --num_labels 8 --batch_size 16 --max_seq_len 128 \
    --valid_steps 50 --logging_steps 10 --num_train_epochs 20 --device "cuda:0"'

# 2. PET
run_task "02_PET" "prompt_tasks/PET/checkpoints/comment_classify/model_best/config.json" \
    bash -c 'cd prompt_tasks/PET && $PYTHON_BIN pet.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/" \
    --img_log_dir "logs/comment_classify" --img_log_name "BERT" \
    --verbalizer "data/comment_classify/verbalizer.txt" \
    --prompt_file "data/comment_classify/prompt.txt" \
    --batch_size 8 --max_seq_len 256 --valid_steps 40 --logging_steps 5 \
    --num_train_epochs 200 --max_label_len 2 --rdrop_coef 5e-2 --device "cuda:0"'

# 3. P-Tuning
run_task "03_p_tuning" "prompt_tasks/p-tuning/checkpoints/comment_classify/model_best/config.json" \
    bash -c 'cd prompt_tasks/p-tuning && $PYTHON_BIN p_tuning.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --verbalizer "data/comment_classify/verbalizer.txt" \
    --save_dir "checkpoints/comment_classify/" \
    --img_log_dir "logs/comment_classify" --img_log_name "BERT" \
    --batch_size 8 --max_seq_len 128 --valid_steps 20 --logging_steps 5 \
    --num_train_epochs 20 --max_label_len 2 --p_embedding_num 15 --device "cuda:0"'

# 4-6. 文本匹配（三种）
run_task "04_matching_pointwise" "text_matching/supervised/checkpoints/comment_classify/pointwise/model_best/model.pt" \
    bash -c 'cd text_matching/supervised && $BASE_PYTHON_BIN train_pointwise.py \
    --model "$MODEL_ROOT/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/pointwise" \
    --img_log_dir "logs/comment_classify" --img_log_name "ERNIE-PointWise" \
    --batch_size 8 --max_seq_len 128 --valid_steps 50 --logging_steps 10 \
    --num_train_epochs 10 --device "cuda:0"'

run_task "05_matching_dssm" "text_matching/supervised/checkpoints/comment_classify/dssm/model_best/model.pt" \
    bash -c 'cd text_matching/supervised && $BASE_PYTHON_BIN train_dssm.py \
    --model "$MODEL_ROOT/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/dssm" \
    --img_log_dir "logs/comment_classify" --img_log_name "ERNIE-DSSM" \
    --batch_size 8 --max_seq_len 128 --valid_steps 50 --logging_steps 10 \
    --num_train_epochs 10 --device "cuda:0"'

run_task "06_matching_sentence_bert" "text_matching/supervised/checkpoints/comment_classify/sentence_transformer/model_best/model.pt" \
    bash -c 'cd text_matching/supervised && $BASE_PYTHON_BIN train_sentence_transformer.py \
    --model "$MODEL_ROOT/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/sentence_transformer" \
    --img_log_dir "logs/comment_classify" --img_log_name "Sentence-Ernie" \
    --batch_size 8 --max_seq_len 256 --valid_steps 50 --logging_steps 10 \
    --num_train_epochs 10 --device "cuda:0"'

# 7. SimCSE（抽样1万条）
head -n 10000 text_matching/unsupervised/simcse/data/LCQMC/train.txt > text_matching/unsupervised/simcse/data/LCQMC/train_subsample.txt
run_task "07_simcse" "text_matching/unsupervised/simcse/checkpoints/LCQMC/model_best/model.pt" \
    bash -c 'cd text_matching/unsupervised/simcse && $BASE_PYTHON_BIN train.py \
    --model "$MODEL_ROOT/ernie-3.0-base-zh" \
    --train_path "data/LCQMC/train_subsample.txt" \
    --dev_path "data/LCQMC/dev.tsv" \
    --save_dir "checkpoints/LCQMC" \
    --img_log_dir "logs/LCQMC" --img_log_name "ERNIE-ESimCSE" \
    --learning_rate 1e-5 --dropout 0.3 --batch_size 64 --max_seq_len 64 \
    --valid_steps 50 --logging_steps 20 --num_train_epochs 2 --device "cuda:0"'

# 8. UIE
run_task "08_UIE" "UIE/checkpoints/DuIE/model_best/model.pt" \
    bash -c 'cd UIE && $BASE_PYTHON_BIN train.py \
    --pretrained_model "$MODEL_ROOT/uie-base-zh" \
    --encoder_model "$MODEL_ROOT/ernie-3.0-base-zh" \
    --save_dir "checkpoints/DuIE" \
    --train_path "data/DuIE/train.txt" --dev_path "data/DuIE/dev.txt" \
    --img_log_dir "logs/" --img_log_name "UIE Base" \
    --batch_size 32 --max_seq_len 256 --learning_rate 5e-5 \
    --num_train_epochs 10 --logging_steps 10 --valid_steps 100 --device "cuda:0"'

# 9. RLHF 奖励模型（抽样1000条rank list）
head -n 1000 RLHF/data/reward_datasets/sentiment_analysis/train.tsv > RLHF/data/reward_datasets/sentiment_analysis/train_subsample.tsv
head -n 200 RLHF/data/reward_datasets/sentiment_analysis/dev.tsv > RLHF/data/reward_datasets/sentiment_analysis/dev_subsample.tsv
run_task "09_rlhf_reward_model" "RLHF/checkpoints/reward_model/sentiment_analysis/model_best/model.pt" \
    bash -c 'cd RLHF && $BASE_PYTHON_BIN train_reward_model.py \
    --model "$MODEL_ROOT/ernie-3.0-base-zh" \
    --train_path "data/reward_datasets/sentiment_analysis/train_subsample.tsv" \
    --dev_path "data/reward_datasets/sentiment_analysis/dev_subsample.tsv" \
    --save_dir "checkpoints/reward_model/sentiment_analysis" \
    --img_log_dir "logs/reward_model/sentiment_analysis" --img_log_name "ERNIE Reward Model" \
    --batch_size 32 --max_seq_len 128 --learning_rate 1e-5 \
    --valid_steps 20 --logging_steps 5 --num_train_epochs 3 --device "cuda:0"'

# 10. RLHF PPO（演示150步）
run_task "10_rlhf_ppo" "RLHF/checkpoints/ppo_sentiment_gpt/final/model.safetensors" \
    bash -c 'cd RLHF && PPO_STEPS=150 $BASE_PYTHON_BIN ppo_sentiment_example.py'

# 11. 问答生成 T5
run_task "11_answer_generation" "answer_generation/checkpoints/DuReaderQG_sentinel_v3/model_best/config.json" \
    bash -c 'cd answer_generation && $BASE_PYTHON_BIN train.py \
    --pretrained_model "$MODEL_ROOT/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/DuReaderQG_sentinel_v3" --target_format sentinel \
    --train_path "data/DuReaderQG/train.json" --dev_path "data/DuReaderQG/dev.json" \
    --img_log_dir "logs/DuReaderQG" --img_log_name "T5-Base-Chinese" \
    --batch_size 32 --max_source_seq_len 256 --max_target_seq_len 32 \
    --learning_rate 5e-5 --num_train_epochs 2 \
    --logging_steps 50 --valid_steps 200 --device "cuda:0"'

# 12. 填空数据增强 T5（抽样5000条）
head -n 5000 data_augment/filling_model/data/train.tsv > data_augment/filling_model/data/train_subsample.tsv
head -n 500 data_augment/filling_model/data/dev.tsv > data_augment/filling_model/data/dev_subsample.tsv
run_task "12_filling_model" "data_augment/filling_model/checkpoints/filling_sentinel_v3/model_best/config.json" \
    bash -c 'cd data_augment/filling_model && $BASE_PYTHON_BIN train.py \
    --pretrained_model "$MODEL_ROOT/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/filling_sentinel_v3" --target_format sentinel \
    --train_path "data/train_subsample.tsv" --dev_path "data/dev_subsample.tsv" \
    --img_log_dir "logs/filling_model" --img_log_name "T5-Filling" \
    --batch_size 32 --max_source_seq_len 256 --max_target_seq_len 64 \
    --learning_rate 5e-5 --num_train_epochs 1 \
    --logging_steps 10 --valid_steps 100 --device "cuda:0"'

# 13-15. LLM zero-shot（需 chatglm-6b）
if [ -f "$MODEL_ROOT/chatglm-6b/config.json" ]; then
    run_task "13_llm_zeroshot_classification" "-" \
        bash -c 'cd LLM/zero-shot && $PYTHON_BIN llm_classification.py'
    run_task "14_llm_zeroshot_matching" "-" \
        bash -c 'cd LLM/zero-shot && $PYTHON_BIN llm_text_matching.py'
    run_task "15_llm_zeroshot_extraction" "-" \
        bash -c 'cd LLM/zero-shot && $PYTHON_BIN llm_information_extraction.py'
else
    echo ">>> [跳过] THUDM/chatglm-6b 未下载，LLM zero-shot 任务跳过"
fi

echo ""
echo "==================== 复现结果汇总 ===================="
cat "$STATUS_FILE"
echo "======================================================"
echo "详细日志: $RESULTS_DIR/*.log"
exit "$RUN_FAILURES"
