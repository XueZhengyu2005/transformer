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

使用T5进行中文问答任务训练，数据集使用百度开源中文问答数据集。

Author: pankeyu
Date: 2023/01/04
"""
import os
import time
import argparse
import json
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, default_data_collator, get_scheduler

from utils import convert_example
sys.path.append(str(Path(__file__).resolve().parents[1]))
from t5_task_format import evaluate_generation
from bleu_metrics import BLEU
from iTrainingLogger import iSummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", default='uer/t5-base-chinese-cluecorpussmall', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_source_seq_len", default=512, type=int,help="The maximum total encoder input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--max_target_seq_len", default=512, type=int,help="The maximum total decoder input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--num_labels", default=2, type=int, help="Total classes of labels.")
parser.add_argument('--target_format', choices=('legacy', 'sentinel'), default='legacy',
                    help='Use UER-T5 extra0/extra1 span format (new checkpoints only).')
parser.add_argument('--max_train_samples', type=int, default=0, help='Optional diagnostic subset.')
parser.add_argument('--max_eval_samples', type=int, default=0, help='Optional diagnostic subset.')
parser.add_argument('--max_train_steps', type=int, default=0, help='Optional diagnostic step budget.')
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, tokenizer, data_loader):
    return evaluate_generation(model, tokenizer, data_loader, args.device,
                               args.max_target_seq_len, 'qa', args.target_format, BLEU)


def save_checkpoint(model, tokenizer, directory):
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    with open(os.path.join(directory, 'task_format.json'), 'w', encoding='utf-8') as handle:
        json.dump({'task': 'qa', 'target_format': args.target_format,
                   'max_source_seq_len': args.max_source_seq_len,
                   'max_target_seq_len': args.max_target_seq_len}, handle,
                  ensure_ascii=False, indent=2)


def train():
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.eos_token = tokenizer.sep_token                               # 因为中文用的是Bert的Tokenizer，所以需要
    tokenizer.bos_token = tokenizer.cls_token                               # 手动指定BOS Token和EOS Token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, 'generation_config'):
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    for split in ('train', 'dev'):
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(
            lambda row: bool(json.loads(row['text'])['answer'].strip()))
        print(f'{split}: removed {before - len(dataset[split])} empty-answer examples')
    convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_source_seq_len=args.max_source_seq_len,
        max_target_seq_len=args.max_target_seq_len,
        target_format=args.target_format,
    )
    if args.max_train_samples:
        dataset['train'] = dataset['train'].shuffle(seed=42).select(
            range(min(args.max_train_samples, len(dataset['train']))))
    if args.max_eval_samples:
        dataset['dev'] = dataset['dev'].select(
            range(min(args.max_eval_samples, len(dataset['dev']))))
    dataset = dataset.map(convert_func, batched=True,
                          load_from_cache_file=(args.target_format == 'legacy'))
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)
    model.train()

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    global_step, best_f1 = 0, -1.0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                **({'decoder_input_ids': batch['decoder_input_ids'].to(args.device)}
                   if 'decoder_input_ids' in batch else {}),
                labels=batch['labels'].to(args.device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                metrics, predictions = evaluate_model(model, tokenizer, eval_dataloader)
                print(f"Evaluation at step {global_step}: "
                      f"char_F1={metrics['char_f1']:.5f}, EM={metrics['exact_match']:.5f}, "
                      f"valid={metrics['valid_rate']:.5f}, BLEU4={metrics['bleu4']:.5f}")
                for item in predictions[:3]:
                    print(f"prediction={item['prediction']!r} reference={item['reference']!r} "
                          f"valid={item['valid']}")
                for name, value in metrics.items():
                    if name != 'count':
                        writer.add_scalar('eval/' + name, value, global_step)
                writer.record()
                os.makedirs(args.save_dir, exist_ok=True)
                with open(os.path.join(args.save_dir, f'predictions_{global_step}.jsonl'),
                          'w', encoding='utf-8') as handle:
                    for item in predictions:
                        handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                if metrics['valid_rate'] > 0 and metrics['char_f1'] > best_f1:
                    best_f1 = metrics['char_f1']
                    save_checkpoint(model, tokenizer, os.path.join(args.save_dir, 'model_best'))
                    with open(os.path.join(args.save_dir, 'best_metrics.json'),
                              'w', encoding='utf-8') as handle:
                        json.dump({'step': global_step, **metrics}, handle,
                                  ensure_ascii=False, indent=2)
                tic_train = time.time()

            if args.max_train_steps and global_step >= args.max_train_steps:
                break
        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    # No intermediate or model_last snapshots: keep only the best reloadable
    # model required by README inference (and the lightweight metrics/logs).
    if best_f1 < 0:
        save_checkpoint(model, tokenizer, os.path.join(args.save_dir, 'model_best'))
        print('No validation checkpoint selected; saved final model_best (unverified).')


if __name__ == '__main__':
    from rich import print
    train()
