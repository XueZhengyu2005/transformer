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
sys.path.append(str(Path(__file__).resolve().parents[1]))
from t5_task_format import qa_source, decode_content

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
max_source_seq_len = 256
checkpoint_path = os.environ.get('MODEL_CHECKPOINT', './checkpoints/DuReaderQG_sentinel_v3/model_best/')
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


def inference(question: str, context: str):
    """
    inference函数。

    Args:
        qustion (str): 问题
        context (str): 原文
    """
    input_seq = qa_source(question, context, target_format, tokenizer, max_source_seq_len)
    inputs = tokenizer(
        text=input_seq,
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
    output, valid = decode_content(tokenizer, outputs[0].cpu().tolist(), 'qa', target_format)
    output = f'答案：{output}' if valid else ''
    print(f'Q: "{question}"')
    print(f'C: "{context}"')
    print(f'A: "{output}"')
    if not valid:
        raise RuntimeError('The QA model did not generate a valid answer.')
    return output


if __name__ == '__main__':
    question = '治疗宫颈糜烂的最佳时间'
    context = '专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。'
    inference(question=question, context=context)
