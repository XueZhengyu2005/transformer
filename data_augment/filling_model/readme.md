## Filling Model with T5

该项目用于将句子中 [MASK] 位置通过生成模型还原，以实现 UIE 信息抽取中 Mask Then Filling [数据增强策略](../../UIE/readme.md)。

[Mask Then Fill](https://arxiv.org/pdf/2301.02427.pdf) 是一种基于生成模型的信息抽取数据增强策略。对于一段文本，我们其分为「关键信息段」和「非关键信息段」，包含关键词片段称为「关键信息段」。下面例子中标粗的为 `关键信息片段`，其余的为 `非关键片段`。

> **大年三十** 我从 **北京** 的大兴机场 **飞回** 了 **成都**。


我们随机 [MASK] 住一部分「非关键片段」，使其变为： 

> **大年三十** 我从 **北京** [MASK] **飞回** 了 **成都**。

随后，将改句子喂给 filling 模型（T5-Fine Tuned）还原句子，得到新生成的句子：

> **大年三十** 我从 **北京** 首都机场作为起点，**飞回** 了 **成都**。

## 1. 环境安装

本项目基于 `pytorch` + `transformers` 实现，运行前请安装相关依赖包：

```sh
pip install -r ../requirements.txt
```

## 2. 数据集准备

项目中提供了一部分示例数据，数据来自DuIE数据集中的文本数据，数据在 `data/` 。

若想使用 `自定义数据` 训练，只需要仿照示例数据构建带 [MASK] 的文本即可，你也可以使用 `parse_data.py` 快速生成基于 `词粒度` 的训练数据：

```tsv
"Bortolaso Guillaume,法国籍[MASK]"中[MASK]位置的文本是：	运动员
"歌曲[MASK]是由歌手海生演唱的一首歌曲"中[MASK]位置的文本是：	《情一动心就痛》
...
```

每一行用 `\t` 分隔符分开，第一部分部分为 `带[MASK]的文本`，后一部分为 `[MASK]位置的原始文本（label）`。


## 3. 模型训练

修改训练脚本 `train.sh` 里的对应参数, 开启模型训练：

```sh
python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/t5" \
    --train_path "data/train.tsv" \
    --dev_path "data/dev.tsv" \
    --img_log_dir "logs" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 128 \
    --max_source_seq_len 128 \
    --max_target_seq_len 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --logging_steps 50 \
    --valid_steps 500 \
    --device cuda:0
```

正确开启训练后，终端会打印以下信息：

```python
...
 0%|          | 0/2 [00:00<?, ?it/s]
100%|██████████| 2/2 [00:00<00:00, 21.28it/s]
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 350134
    })
    dev: Dataset({
        features: ['text'],
        num_rows: 38904
    })
})
...
global step 2400, epoch: 1, loss: 7.44746, speed: 0.82 step/s
global step 2450, epoch: 1, loss: 7.42028, speed: 0.82 step/s
global step 2500, epoch: 1, loss: 7.39333, speed: 0.82 step/s
Evaluation bleu4: 0.00578
best BLEU-4 performence has been updated: 0.00026 --> 0.00578
global step 2550, epoch: 1, loss: 7.36620, speed: 0.81 step/s
...
```

在 `logs/T5-Base-Chinese.png` 文件中将会保存训练曲线图：

</img>

## 4. 模型预测

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```python
 if __name__ == "__main__":
    masked_texts = [
        '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
    ]
    inference(masked_texts)
```

```sh
python inference.py
```

得到以下推理结果：

```python
maksed text: 
[
    '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
]
output: 
[
    '，中国工业出版社出版的'
]
```
## 复现与验收补充（2026-09-29）

原始目标是给含有 `[MASK]` 的文本补出能够放回原句的片段，而不是仅仅生成文本。原项目的旧模型及其日志不覆盖；以下使用 UER 中文 T5 已经预训练的 `extra0`/`extra1` 哨兵格式，新旧 checkpoint 通过 `task_format.json` 区分，推理接口仍接受 README 中带 `[MASK]` 的原始输入。

```sh
cd data_augment/filling_model
python train.py \
  --pretrained_model /root/autodl-tmp/upload_models/t5-base-chinese-cluecorpussmall \
  --train_path data/train.tsv --dev_path data/dev.tsv \
  --save_dir checkpoints/filling_sentinel_new \
  --img_log_dir logs/filling_sentinel_new --img_log_name T5-Filling-Sentinel \
  --target_format sentinel --max_source_seq_len 256 --max_target_seq_len 64 \
  --batch_size 16 --learning_rate 1e-4 --num_train_epochs 3 \
  --valid_steps 500 --logging_steps 100 --device cuda:0
```

完整原始训练集有约 35 万行，上述命令需按 GPU 时长安排；资源受限时可以加 `--max_train_samples 5000 --max_eval_samples 128 --max_train_steps 900` 做**诊断**，不能把子集结果当成全量复现。内容级字符 F1 负责选择 `model_best`；新训练仅保留这一份供推理使用的最终权重，不再产生按步快照或 `model_last`。用完整 dev 和 README 演示样例核验：

```sh
cd ../..
python tools/verify_t5.py fill data_augment/filling_model/checkpoints/filling_sentinel_new/model_best \
  --dev-path data_augment/filling_model/data/dev_subsample.tsv \
  --output reproduce_results/filling_sentinel_subsample
cd data_augment/filling_model
MODEL_CHECKPOINT=checkpoints/filling_sentinel_new/model_best python inference.py
```

生成片段需能放回原句，并保存逐样本预测供人工审查。BLEU、字符 F1 和精确匹配仅为参考：存在多个合理补法，也可能有对原句造成时间/语义矛盾的预测。现有 5,000/500 顺序子集里，按填回片段恢复原文有 8 条 dev 与训练原文重复；把此结果作为实验进展而非独立泛化验收。完整数据实验需重新检查数据分组与重叠。

若需评价 README 中全部 38,904 条原始 dev，去掉 `--dev-path`；当前 5,000/500 子集实验只在显式指定的 500 条 dev 子集上做对应评价，完整 dev 推理耗时很长，不可将子集结果冒充全量结果。

当前已重载验证的最好探索权重为 `checkpoints/filling_sentinel_v3/model_best`，默认 `inference.py` 指向它。对本次使用的 500 条 dev 子集，排除 4 条空答案和 7 条在早期 warm-start 阶段已经见过恢复原文的样本后，489 条字符 F1=`0.22545`、精确匹配=`0.02863`、有效输出率=`0.94888`，详见 `reproduce_results/12_filling_sentinel_v3_clean.json` 和 `.jsonl`。验证命令须带 `--dev-path data_augment/filling_model/data/dev_subsample.tsv --exclude-originals-from data_augment/filling_model/data/train_subsample.tsv`。当前演示例子生成“出版的”，可语法补句但不够具体，因此**仍未达到稳定正确的填空验收**。其 20,000 行子集后两阶段训练不等于 README 原始全量训练。受模型权重和数据缓存占用影响，继续全量训练前应先规划磁盘空间。
