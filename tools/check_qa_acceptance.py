"""Check that the saved full-dev QA result still matches the actual checkpoint."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
summary_path = root / 'reproduce_results/11_qa_sentinel_v3_full.json'
checkpoint = root / 'answer_generation/checkpoints/DuReaderQG_sentinel_v3/model_best'
try:
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    weight = checkpoint / 'model.safetensors'
    sha = hashlib.sha256()
    with weight.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    assert summary['checkpoint'].replace('\\', '/').endswith('/DuReaderQG_sentinel_v3/model_best')
    assert summary['checkpoint_sha256'] == sha.hexdigest()
    assert summary['metrics']['count'] == 983
    assert summary['demo']['valid'] and summary['demo']['prediction'] == summary['demo']['reference']
except (OSError, ValueError, KeyError, AssertionError) as error:
    print(f'QA validation missing or stale: {error}', file=sys.stderr)
    raise SystemExit(1)
print('QA validated: full dev metrics, exact README demo, matching checkpoint SHA256')
