"""Compute compact category vectors needed by DSSM/SentenceTransformer demos.

This is a derived JSON cache, NOT a model checkpoint.  Its contents should be
regenerated when the corresponding model_best weights or type descriptions change.
"""
import json
from pathlib import Path

import torch


def load_or_build(model, tokenizer, device, model_type, output_file):
    output_file = Path(output_file)
    model_file = Path('checkpoints/comment_classify') / model_type / 'model_best/model.pt'
    source = Path('data/comment_classify/types_desc.txt')
    if output_file.exists() and output_file.stat().st_mtime >= max(model_file.stat().st_mtime,
                                                                    source.stat().st_mtime):
        with output_file.open(encoding='utf-8') as handle:
            return json.load(handle)

    records = {}
    model.eval()
    with source.open(encoding='utf-8') as handle, torch.no_grad():
        for line in handle:
            fields = line.rstrip('\n').split('\t', 2)
            if len(fields) not in (2, 3):
                raise ValueError(f'Unexpected category description: {line!r}')
            key, label = fields[:2]
            text = label + '：' + (fields[2] if len(fields) == 3 else '')
            batch = tokenizer(text, truncation=True, max_length=256,
                              padding='max_length', return_tensors='pt').to(device)
            inputs = dict(input_ids=batch['input_ids'],
                          token_type_ids=batch.get('token_type_ids'),
                          attention_mask=batch['attention_mask'])
            if model_type == 'dssm':
                embedding = model(**inputs)
            elif model_type == 'sentence_transformer':
                embedding = model.get_embedding(**inputs)
            else:
                raise ValueError(f'Unsupported model type: {model_type}')
            records[key] = {'label': label, 'text': text,
                            'embedding': embedding.squeeze(0).detach().cpu().tolist()}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as handle:
        json.dump(records, handle, ensure_ascii=False)
    return records
