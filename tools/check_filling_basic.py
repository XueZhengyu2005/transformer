"""Verify the saved filling demo, not the general accuracy of the model."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reproduce_results/12_filling_sentinel_v3_clean.json'
MODEL = ROOT / 'data_augment/filling_model/checkpoints/filling_sentinel_v3/model_best'
try:
    info = json.loads(REPORT.read_text(encoding='utf-8'))
    digest = hashlib.sha256()
    with (MODEL / 'model.safetensors').open('rb') as weights:
        for chunk in iter(lambda: weights.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    example = info['demo']
    completed = example['source'].replace(' extra0 ', example['prediction'], 1)
    assert info['checkpoint'].replace('\\', '/').endswith('/filling_sentinel_v3/model_best')
    assert digest.hexdigest() == info['checkpoint_sha256']
    assert example['valid'] and '2005年2月出版的图书' in completed
except (OSError, KeyError, AssertionError, ValueError) as exc:
    print(f'Filling demonstration is missing or invalid: {exc}', file=sys.stderr)
    raise SystemExit(1)
print('Filling README example works; this is NOT a generalization or exact-match claim.')
