"""Input/target contract shared by the two T5 generation examples.

The UER Chinese T5 backbone was pretrained with ``extra0``/``extra1``
span sentinels.  Legacy examples keep their original prompts; new sentinel
runs have an explicit, separately saved format so inference cannot silently
mix the two incompatible protocols.
"""
from collections import Counter

SENTINEL_START = 'extra0'
SENTINEL_END = 'extra1'
FORMATS = ('legacy', 'sentinel')


def qa_source(question, context, target_format='legacy', tokenizer=None,
              max_length=None, answer=None):
    if target_format not in FORMATS:
        raise ValueError(f'Unknown format: {target_format}')
    # Deliberately do not crop using `answer`: the answer is unavailable at
    # inference time. Oracle answer-aware validation would inflate metrics.
    prefix = f'问题：{question}[SEP]原文：'
    suffix = ' ' + SENTINEL_START if target_format == 'sentinel' else ''
    if tokenizer is not None and max_length is not None and suffix:
        # Reserve room for the sentinel and the tokenizer's wrapper tokens;
        # otherwise truncation silently drops extra0 on long articles.
        low, high = 0, len(context)
        while low < high:
            mid = (low + high + 1) // 2
            if len(tokenizer.encode(prefix + context[:mid] + suffix)) <= max_length:
                low = mid
            else:
                high = mid - 1
        context = context[:low]
    return prefix + context + suffix


def fill_source(prompt, target_format='legacy'):
    if target_format == 'legacy':
        return prompt
    if target_format != 'sentinel':
        raise ValueError(f'Unknown format: {target_format}')
    # Original dataset uses: '"sentence[MASK]"中[MASK]位置的文本是：'.
    # Only the mask inside the quoted sentence should be replaced.
    sentence, separator, _ = prompt.partition('"中[MASK]位置的文本是：')
    if not separator or not sentence.startswith('"') or sentence.count('[MASK]') != 1:
        raise ValueError(f'Unexpected filling prompt: {prompt[:100]!r}')
    return sentence[1:].replace('[MASK]', ' ' + SENTINEL_START + ' ', 1)


def reconstruct_fill_original(raw):
    """Recover the unmasked original sentence for group-safe splitting."""
    prompt, answer = raw.rstrip('\n').split('\t', 1)
    sentence, separator, _ = prompt.partition('"中[MASK]位置的文本是：')
    if not separator or not sentence.startswith('"') or sentence.count('[MASK]') != 1:
        raise ValueError(f'Unexpected filling prompt: {prompt[:100]!r}')
    return sentence[1:].replace('[MASK]', answer, 1)


def target_text(answer, task, target_format='legacy'):
    if target_format not in FORMATS or task not in ('qa', 'fill'):
        raise ValueError(f'Unsupported T5 task/format: {task}/{target_format}')
    text = ('答案：' if task == 'qa' else '') + answer
    return f'{SENTINEL_START}{text}{SENTINEL_END}' if target_format == 'sentinel' else text


def make_labels(tokenizer, answer, task, target_format, max_target_seq_len):
    """Exactly one EOS, with ignored loss on padding; T5 shifts labels itself."""
    if max_target_seq_len < 4:
        raise ValueError('max_target_seq_len must be at least 4')
    if target_format == 'sentinel':
        # BertTokenizer can merge "extra0" with adjacent Latin letters into
        # [UNK]; insert sentinel token IDs explicitly, never as concatenated
        # free text.  Keep room for both sentinels and EOS when truncating.
        first = tokenizer.convert_tokens_to_ids(SENTINEL_START)
        last = tokenizer.convert_tokens_to_ids(SENTINEL_END)
        if first == tokenizer.unk_token_id or last == tokenizer.unk_token_id:
            raise ValueError('Tokenizer lacks the UER T5 sentinel tokens')
        content = ('答案：' if task == 'qa' else '') + answer
        body = tokenizer.encode(content, add_special_tokens=False,
                                truncation=True, max_length=max_target_seq_len - 3)
        ids = [first] + body + [last, tokenizer.eos_token_id]
    else:
        ids = tokenizer.encode(target_text(answer, task, target_format),
                               add_special_tokens=False, truncation=True,
                               max_length=max_target_seq_len - 1)
        ids.append(tokenizer.eos_token_id)
    return ids + [-100] * (max_target_seq_len - len(ids))


def decode_content(tokenizer, ids, task, target_format, allow_unknown=False):
    """Return (content, format_valid); a sentinel-only output is invalid."""
    seq = [int(x) for x in ids if int(x) >= 0 and int(x) != tokenizer.pad_token_id]
    sentinel_ids = {tokenizer.convert_tokens_to_ids(f'extra{i}') for i in range(100)}
    if target_format == 'sentinel':
        first = tokenizer.convert_tokens_to_ids(SENTINEL_START)
        last = tokenizer.convert_tokens_to_ids(SENTINEL_END)
        if first not in seq or last not in seq or seq.index(first) >= seq.index(last):
            return '', False
        # A start token in the body signals malformed output, not a valid span.
        start, stop = seq.index(first), seq.index(last)
        if any(token_id in sentinel_ids for token_id in seq[start + 1:stop]):
            return '', False
        seq = seq[start + 1:stop]
    else:
        if any(token_id in sentinel_ids for token_id in seq):
            return '', False
        seq = [x for x in seq if x not in {tokenizer.cls_token_id, tokenizer.sep_token_id,
                                          tokenizer.eos_token_id}]
    content = tokenizer.decode(seq, skip_special_tokens=not allow_unknown).replace(' ', '').strip()
    if task == 'qa' and content.startswith('答案：'):
        content = content[len('答案：'):]
    if not content or (not allow_unknown and '[UNK]' in content) or '[MASK]' in content:
        return '', False
    return content, True


def char_f1(prediction, reference):
    predicted, expected = Counter(prediction), Counter(reference)
    shared = sum((predicted & expected).values())
    return 2 * shared / (len(prediction) + len(reference)) if prediction and reference else 0.0


def evaluate_generation(model, tokenizer, data_loader, device, max_new_tokens,
                        task, target_format, bleu_class):
    """Score *generated content*, never loss-padding or task delimiters."""
    import torch

    bleu = [bleu_class(n_size=n) for n in range(1, 5)]
    scores, exact, valid_count, predictions = [], 0, 0, []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in data_loader:
                output = model.generate(input_ids=batch['input_ids'].to(device),
                                        attention_mask=batch['attention_mask'].to(device),
                                        max_new_tokens=max_new_tokens,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
                for batch_index, (prediction_ids, reference_ids) in enumerate(
                        zip(output.cpu().tolist(), batch['labels'].tolist())):
                    prediction, valid = decode_content(tokenizer, prediction_ids, task, target_format)
                    reference, reference_valid = decode_content(tokenizer, reference_ids, task, target_format, allow_unknown=True)
                    if not reference_valid:
                        raise ValueError(f'Invalid validation target: {reference_ids}')
                    # Independent verification can supply the untruncated raw
                    # answer; labels may be cut at max_target_seq_len or use UNK.
                    raw_reference = batch.get('reference_text')
                    if raw_reference is not None:
                        reference = raw_reference[batch_index]
                    scores.append(char_f1(prediction, reference) if valid else 0.0)
                    exact += int(valid and prediction == reference)
                    valid_count += int(valid)
                    predictions.append({'prediction': prediction, 'reference': reference,
                                        'valid': valid, 'raw': tokenizer.decode(prediction_ids)})
                    pred_tokens = tokenizer.encode(prediction, add_special_tokens=False) if valid else [-1]
                    ref_tokens = tokenizer.encode(reference, add_special_tokens=False)
                    for metric in bleu:
                        metric.add_instance(prediction=pred_tokens, references=[ref_tokens])
    finally:
        model.train(was_training)
    total = len(scores)
    if not total:
        raise ValueError('Empty validation set')
    return ({'bleu1': bleu[0].compute(), 'bleu2': bleu[1].compute(),
             'bleu3': bleu[2].compute(), 'bleu4': bleu[3].compute(),
             'char_f1': sum(scores) / total, 'exact_match': exact / total,
             'valid_rate': valid_count / total, 'count': total}, predictions)
