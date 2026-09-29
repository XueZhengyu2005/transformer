# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

测试训练好的模型效果。

Author: pankeyu
Date: 2023/01/05
"""
from rich import print
import os
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
sys.path.append(str(Path(__file__).resolve().parents[2]))
from t5_task_format import fill_source, decode_content

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
max_source_seq_len = 128
checkpoint_path = os.environ.get('MODEL_CHECKPOINT', './checkpoints/filling_sentinel_v3/model_best/')
format_file = os.path.join(checkpoint_path, 'task_format.json')
task_config = {}
if os.path.isfile(format_file):
    with open(format_file, encoding='utf-8') as handle:
        task_config = json.load(handle)
target_format = task_config.get('target_format', 'legacy')
max_source_seq_len = task_config.get('max_source_seq_len', max_source_seq_len)
max_target_seq_len = task_config.get('max_target_seq_len', 32 if target_format == 'legacy' else 64)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.eos_token = tokenizer.sep_token
tokenizer.bos_token = tokenizer.cls_token
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
if hasattr(model, 'generation_config'):
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
model.to(device).eval()


def inference(masked_texts: list):
    """
    inference函数。

    Args:
        masked_texts (list): 掩码后的文字列表
    """
    inputs = tokenizer(
        text=[fill_source(text, target_format) for text in masked_texts],
        truncation=True,
        max_length=max_source_seq_len,
        padding='max_length',
        return_tensors='pt'
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=max_target_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = [decode_content(tokenizer, output.cpu().tolist(), 'fill', target_format)
               for output in outputs]
    outputs = [content for content, _ in decoded]
    print(f'maksed text: {masked_texts}')
    print(f'output: {outputs}')
    if not all(valid for _, valid in decoded):
        raise RuntimeError('The filling model did not generate a valid span.')
    return outputs


if __name__ == '__main__':
    masked_texts = [
        '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
    ]
    inference(masked_texts)
