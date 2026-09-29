"""Independent checkpoint validation for the two README T5 examples.

Examples:
  python tools/verify_t5.py qa answer_generation/checkpoints/DuReaderQG_sentinel_v1/model_best
  python tools/verify_t5.py fill data_augment/filling_model/checkpoints/filling_sentinel_v1/model_best
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t5_task_format import (char_f1, decode_content, evaluate_generation,
                            fill_source, qa_source, reconstruct_fill_original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('qa', 'fill'))
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--dev-path', type=Path,
                        help='Explicit evaluation split; default is the full original dev set')
    parser.add_argument('--exclude-originals-from', type=Path,
                        help='Filling: discard dev originals seen in a prior warm-start run')
    parser.add_argument('--limit', type=int, default=0,
                        help='0 evaluates all rows in the selected dev set')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path,
                        help='Save summary JSON + all predictions as JSONL; never overwrite the old logs')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = checkpoint / 'task_format.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'{config_path}: this checkpoint predates explicit format metadata')
    with config_path.open(encoding='utf-8') as handle:
        config = json.load(handle)
    if config['task'] != args.task:
        raise ValueError(f"Wrong checkpoint task: expected {args.task}, got {config['task']}")
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True)
    model.to(args.device).eval()
    dev_path = (args.dev_path or (ROOT / 'answer_generation/data/DuReaderQG/dev.json'
                                  if args.task == 'qa' else
                                  ROOT / 'data_augment/filling_model/data/dev.tsv')).resolve()
    prior_originals = set()
    if args.exclude_originals_from:
        if args.task != 'fill':
            parser.error('--exclude-originals-from only applies to filling')
        with args.exclude_originals_from.open(encoding='utf-8') as previous:
            prior_originals = {reconstruct_fill_original(raw) for raw in previous
                               if raw.rstrip('\n').split('\t', 1)[1].strip()}
    excluded = 0
    samples = []
    with dev_path.open(encoding='utf-8') as handle:
        for raw in handle:
            if args.task == 'qa':
                obj = json.loads(raw)
                if not obj['answer'].strip():
                    continue
                samples.append((qa_source(obj['question'], obj['context'], config['target_format'],
                                          tok, config['max_source_seq_len']), obj['answer']))
            else:
                prompt, answer = raw.rstrip('\n').split('\t', 1)
                if not answer.strip():
                    continue
                if prior_originals and reconstruct_fill_original(raw) in prior_originals:
                    excluded += 1
                    continue
                samples.append((fill_source(prompt, config['target_format']), answer))
            if args.limit and len(samples) >= args.limit:
                break
    from torch.utils.data import DataLoader
    from answer_generation.bleu_metrics import BLEU
    from t5_task_format import make_labels

    def collate(batch):
        enc = tok([s for s, _ in batch], max_length=config['max_source_seq_len'],
                  padding='max_length', truncation=True, return_tensors='pt')
        labels = [make_labels(tok, answer, args.task, config['target_format'],
                              config['max_target_seq_len']) for _, answer in batch]
        return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
                'labels': torch.tensor(labels), 'reference_text': [answer for _, answer in batch]}

    metrics, predictions = evaluate_generation(
        model, tok, DataLoader(samples, batch_size=args.batch_size, collate_fn=collate),
        args.device, config['max_target_seq_len'], args.task, config['target_format'], BLEU)
    for item, (source, _) in zip(predictions, samples):
        item['source'] = source
    # README examples are demonstrations, not validation data; report separately.
    if args.task == 'qa':
        demo_reference = '月经干净后3-7日'
        demo = qa_source('治疗宫颈糜烂的最佳时间',
                         '专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。',
                         config['target_format'], tok, config['max_source_seq_len'])
    else:
        demo_reference = '由科学出版社出版的'  # dev.tsv first example
        demo = fill_source('"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：', config['target_format'])
    encoded = tok(demo, max_length=config['max_source_seq_len'], truncation=True,
                  return_tensors='pt').to(args.device)
    with torch.no_grad():
        demo_ids = model.generate(input_ids=encoded['input_ids'],
                                  attention_mask=encoded['attention_mask'], max_new_tokens=config['max_target_seq_len'],
                                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)[0].tolist()
    demo_content, demo_valid = decode_content(tok, demo_ids, args.task, config['target_format'])
    weights = checkpoint / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'Expected model weights: {weights}')
    digest = hashlib.sha256()
    with weights.open('rb') as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    summary = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest.hexdigest(),
               'dev_path': str(dev_path),
               'excluded_prior_train_originals': excluded, 'metrics': metrics,
               'demo': {'source': demo, 'prediction': demo_content, 'valid': demo_valid,
                        'reference': demo_reference,
                        'char_f1': char_f1(demo_content, demo_reference) if demo_valid else 0.0,
                        'raw': tok.decode(demo_ids)}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        with args.output.with_suffix('.jsonl').open('w', encoding='utf-8') as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
