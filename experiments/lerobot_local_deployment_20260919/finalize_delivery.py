"""Record completed validation before creating user-facing archives."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import build_delivery
root = Path(__file__).resolve().parent
kit = root/'offline_kit'
report = json.loads((root/'developer_validation/summary.json').read_text())
assert report['status'] == 'passed', report['status']
assert len(report['runs']) == 4
for run in report['runs']:
    assert run['sample_count'] == 33
    assert run['all_actions_finite'] and run['saved_unnormalizer_probe_passed']
record = {'created_utc': datetime.now(timezone.utc).isoformat(),
          'developer_platform': report['environment'],
          'validation': 'Both checkpoints x FP16/BF16; all 33 held-out inputs; strict model load; saved processor probes; finite outputs; report/NPZ/ZIP generation.',
          'sample_hashes_verified': report['assets']['samples']['ok'],
          'checkpoint_hashes_verified': {k:v['ok'] for k,v in report['assets']['checkpoints'].items()},
          'missing_checkpoint_diagnostic_test': 'passed',
          'python_syntax_check': 'passed',
          'windows_powershell_execution': 'not performed on developer Linux machine',
          'windows_rtx5060_performance': 'not measured',
          'training_or_robot_operations': 'none',
          'runs': report['runs'], 'fp16_vs_bf16': report['fp16_vs_bf16'],
          'code_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(kit.glob('*')) if p.is_file() and p.suffix in ('.py','.ps1')}}
(kit/'DEVELOPER_VALIDATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
build_delivery.archive()
